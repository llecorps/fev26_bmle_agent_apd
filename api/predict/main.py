"""
APD France — API de prédiction + monitoring.

Sert le modèle RandomForest (cible = log1p(Engagements K EUR)) via /predict,
et expose des métriques Prometheus (/metrics) alimentées par /evaluate :
performances du modèle (RMSE/MAE/R²) et dérive des données (Evidently).

Le modèle est chargé depuis MLflow Registry (alias "champion") avec fallback local.
"""
import json
import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

# ─── Config ──────────────────────────────────────
MODEL_DIR = Path("/app/model")
MONITORING_DIR = Path(os.getenv("MONITORING_DIR", "/app/monitoring"))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "apd-randomforest")
CHAMPION_ALIAS = "champion"
TARGET = "Engagements (K EUR)"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apd-api")

# ─── Registre & métriques Prometheus ─────────────────────────────────────────
# Registre dédié (au lieu du REGISTRY global) : contrôle explicite de ce qu'on
# expose et pas de collision avec d'éventuelles métriques par défaut.
REGISTRY = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total", "Nombre total de requêtes HTTP reçues par l'API",
    ["endpoint", "method", "status_code"], registry=REGISTRY,
)
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds", "Durée des requêtes HTTP (secondes)",
    ["endpoint", "method", "status_code"], registry=REGISTRY,
)

# Scores du modèle de régression, mis à jour par /evaluate.
model_rmse_score = Gauge("model_rmse_score", "RMSE du modèle (échelle log1p)", registry=REGISTRY)
model_mae_score = Gauge("model_mae_score", "MAE du modèle (échelle log1p)", registry=REGISTRY)
model_r2_score = Gauge("model_r2_score", "R² du modèle (échelle log1p)", registry=REGISTRY)

# ── Métrique personnalisée : part de features en dérive (Evidently) ──
# Justification : sur un modèle de régression en production, les vraies étiquettes
# (montants réellement engagés) n'arrivent qu'avec retard. La DÉRIVE DES FEATURES
# est donc l'indicateur AVANCÉ le plus utile : elle prévient d'une dégradation
# probable des performances AVANT même de pouvoir recalculer le RMSE sur des
# données fraîches labellisées. On suit la part de colonnes en dérive [0..1] et
# un statut binaire de dérive globale du dataset.
model_dataset_drift_share = Gauge(
    "model_dataset_drift_share", "Part des features en dérive (0..1) — Evidently", registry=REGISTRY,
)
evidently_data_drift_detected_status = Gauge(
    "evidently_data_drift_detected_status", "Dérive globale du dataset détectée (1) ou non (0)",
    registry=REGISTRY,
)

# ─── State ───────────────────────────────────────
pipeline = None
meta = None
dropdowns = None
model_source = "none"


def load_from_mlflow():
    """Charge le modèle champion depuis MLflow Registry (MLflow 3 aliases)."""
    global pipeline, meta, dropdowns, model_source
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    run_id, version = mv.run_id, mv.version
    logger.info(f"MLflow : {MODEL_NAME} v{version} (alias={CHAMPION_ALIAS}, run={run_id})")

    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")

    artifacts_dir = client.download_artifacts(run_id, "")
    meta_path = Path(artifacts_dir) / "meta.json"
    drop_path = Path(artifacts_dir) / "dropdowns.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    if drop_path.exists():
        dropdowns = json.loads(drop_path.read_text())

    model_source = f"mlflow:{MODEL_NAME}/v{version}@{CHAMPION_ALIAS}"


def load_from_local():
    """Fallback : charge depuis le volume monté /app/model."""
    global pipeline, meta, dropdowns, model_source
    pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    with open(MODEL_DIR / "dropdowns.json") as f:
        dropdowns = json.load(f)
    model_source = "local:/app/model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔄 Chargement du modèle...")
    try:
        load_from_mlflow()
        logger.info(f"✅ Modèle chargé depuis MLflow ({model_source})")
    except Exception as e:
        logger.warning(f"⚠️  MLflow indisponible ({e}), fallback local...")
        try:
            load_from_local()
            logger.info(f"✅ Modèle chargé en local ({model_source})")
        except Exception as e2:
            logger.error(f"❌ Aucun modèle disponible : {e2}")
    yield


app = FastAPI(
    title="APD France — Prediction & Monitoring API",
    version="4.0.0",
    description="Prédiction du montant d'engagement APD + métriques Prometheus/Evidently",
    lifespan=lifespan,
)


# ─── Middleware d'instrumentation ────────────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Compte chaque requête et mesure sa latence, par endpoint/méthode/statut."""
    start = time.perf_counter()
    # Template de route (ex: /predict) plutôt que l'URL brute -> faible cardinalité.
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        labels = dict(endpoint=endpoint, method=request.method, status_code=str(status))
        api_requests_total.labels(**labels).inc()
        api_request_duration_seconds.labels(**labels).observe(elapsed)
    return response


# ─── Schemas ─────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict = Field(
        ...,
        examples=[{
            "Agence": "Agence française de développement",
            "Type de financement": "Prêt",
            "Pays beneficiaire": "Egypte",
            "Catégorie CAD": "PRITI",
        }],
    )

class PredictResponse(BaseModel):
    log_prediction: float
    montant_keur: float
    montant_label: str
    tranche: str
    fourchette_keur: dict
    features_used: dict
    model_source: str

class ModelInfo(BaseModel):
    model_type: str
    model_source: str
    n_features: int
    metrics: dict
    train_size: int
    test_size: int

class EvaluateRequest(BaseModel):
    """Données courantes optionnelles. Si absentes, on charge le parquet de
    février (current_february.parquet). Chaque enregistrement doit contenir les
    features du modèle + la cible réelle `Engagements (K EUR)`."""
    current_data: list[dict] | None = Field(
        default=None,
        description="Enregistrements courants (features + 'Engagements (K EUR)').",
    )
    period_label: str = Field(default="february", description="Libellé de la période courante.")

class EvaluationReportOutput(BaseModel):
    reference_period: str
    current_period: str
    n_reference: int
    n_current: int
    rmse: float
    mae: float
    r2: float
    dataset_drift_detected: bool
    dataset_drift_share: float


# ─── Endpoints ───────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None, "model_source": model_source}

@app.get("/metrics")
def metrics():
    """Expose toutes les métriques Prometheus du registre dédié."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info", response_model=ModelInfo)
def model_info():
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_type="RandomForestRegressor",
        model_source=model_source,
        n_features=len(meta["feature_cols"]),
        metrics={"rmse_log": meta["rmse"], "mae_log": meta["mae"], "r2_log": meta["r2"]},
        train_size=meta["train_size"],
        test_size=meta["test_size"],
    )

@app.get("/model/features")
def model_features():
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "numerical": meta["num_cols"],
        "categorical_low": meta.get("cat_low", []),
        "categorical_high": meta.get("cat_high", []),
        "dropdowns": dropdowns or {},
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = {}
    for col in meta["feature_cols"]:
        if col in req.features and req.features[col] is not None:
            row[col] = req.features[col]
        elif col in meta["num_cols"]:
            row[col] = 0.0
        else:
            row[col] = None

    X = pd.DataFrame([row])[meta["feature_cols"]]
    log_pred = float(pipeline.predict(X)[0])
    montant_keur = float(np.expm1(log_pred))

    Q_BOUNDS = [1.792, 2.852, 5.017]
    if log_pred <= Q_BOUNDS[0]:
        tranche = "Petit (≤ 5 K EUR)"
    elif log_pred <= Q_BOUNDS[1]:
        tranche = "Moyen (5 – 16 K EUR)"
    elif log_pred <= Q_BOUNDS[2]:
        tranche = "Grand (16 – 150 K EUR)"
    else:
        tranche = "Très grand (> 150 K EUR)"

    rmse = meta["rmse"]
    low = float(np.expm1(log_pred - rmse))
    high = float(np.expm1(log_pred + rmse))
    label = f"{montant_keur:,.0f} K EUR" if montant_keur < 1000 else f"{montant_keur / 1000:,.1f} M EUR"

    return PredictResponse(
        log_prediction=round(log_pred, 3),
        montant_keur=round(montant_keur, 1),
        montant_label=label,
        tranche=tranche,
        fourchette_keur={"low": round(low, 1), "high": round(high, 1)},
        features_used={k: v for k, v in req.features.items() if v is not None},
        model_source=model_source,
    )


# ─── /evaluate : performances + dérive ───────────────────────────────────────
def _load_reference() -> pd.DataFrame:
    path = MONITORING_DIR / "reference_january.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Référence absente : {path} "
                            "(lancer `make monitoring-data`)")
    return pd.read_parquet(path)


def _load_current(req: EvaluateRequest) -> pd.DataFrame:
    if req.current_data:
        return pd.DataFrame(req.current_data)
    path = MONITORING_DIR / "current_february.parquet"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Données courantes absentes : {path}")
    return pd.read_parquet(path)


def _regression_scores(df: pd.DataFrame) -> tuple[float, float, float]:
    """Prédit puis calcule RMSE/MAE/R² (échelle log1p) sur un jeu labellisé."""
    feats = meta["feature_cols"]
    X = df[feats]
    y_true = np.log1p(df[TARGET].astype(float))
    y_pred = pipeline.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _drift(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[bool, float]:
    """Dérive des features via Evidently (DataDriftPreset). Fallback KS/PSI simple
    si l'API Evidently diffère selon la version installée."""
    feats = meta["feature_cols"]
    ref, cur = reference[feats], current[feats]
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        result = report.as_dict()["metrics"][0]["result"]
        detected = bool(result.get("dataset_drift", False))
        share = float(result.get("share_of_drifted_columns", 0.0))
        return detected, share
    except Exception as e:  # pragma: no cover - dépend de la version d'Evidently
        logger.warning(f"Evidently indisponible ({e}) — fallback dérive KS.")
        from scipy.stats import ks_2samp
        num = [c for c in feats if pd.api.types.is_numeric_dtype(ref[c])]
        drifted = sum(1 for c in num if ks_2samp(ref[c].dropna(), cur[c].dropna()).pvalue < 0.05)
        share = drifted / len(feats) if feats else 0.0
        return share > 0.5, share


@app.post("/evaluate", response_model=EvaluationReportOutput)
def evaluate(req: EvaluateRequest):
    """Évalue le modèle sur des données courantes vs la référence (janvier) et
    met à jour les Gauges Prometheus (RMSE/MAE/R² + dérive)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    reference = _load_reference()
    current = _load_current(req)

    rmse, mae, r2 = _regression_scores(current)
    detected, share = _drift(reference, current)

    # Mise à jour des métriques exposées à Prometheus.
    model_rmse_score.set(rmse)
    model_mae_score.set(mae)
    model_r2_score.set(r2)
    model_dataset_drift_share.set(share)
    evidently_data_drift_detected_status.set(1 if detected else 0)

    return EvaluationReportOutput(
        reference_period="january",
        current_period=req.period_label,
        n_reference=len(reference),
        n_current=len(current),
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(r2, 4),
        dataset_drift_detected=detected,
        dataset_drift_share=round(share, 4),
    )
