"""
APD France — API de prédiction
Sert le modèle RandomForest via /predict
le modèle est chargé  depuis MLflow Registry (alias "champion") avec fallback local.
"""
import json
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─── Config ──────────────────────────────────────
MODEL_DIR = Path("/app/model")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "apd-randomforest")
CHAMPION_ALIAS = "champion"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("apd-api")

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

    # Récupérer la version avec l'alias "champion"
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    run_id = mv.run_id
    version = mv.version
    logger.info(f"MLflow : {MODEL_NAME} v{version} (alias={CHAMPION_ALIAS}, run={run_id})")

    # Charger le pipeline sklearn
    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Charger meta.json et dropdowns.json depuis les artefacts du run
    artifacts_dir = client.download_artifacts(run_id, "")
    meta_path = Path(artifacts_dir) / "meta.json"
    drop_path = Path(artifacts_dir) / "dropdowns.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    if drop_path.exists():
        with open(drop_path) as f:
            dropdowns = json.load(f)

    model_source = f"mlflow:{MODEL_NAME}/v{version}@{CHAMPION_ALIAS}"


def load_from_local():
    """Fallback : charge depuis le volume monté /app/model."""
    global pipeline, meta, dropdowns, model_source

    pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "dropdowns.json") as f:
        dropdowns = json.load(f)

    model_source = "local:/app/model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, meta, dropdowns, model_source

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
    title="APD France — Prediction API",
    version="3.0.0",
    description="Prédiction du montant d'engagement APD — modèle servi via MLflow 3",
    lifespan=lifespan,
)


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


# ─── Endpoints ───────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_source": model_source,
    }

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
        "categorical_low": meta["cat_low"],
        "categorical_high": meta["cat_high"],
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
