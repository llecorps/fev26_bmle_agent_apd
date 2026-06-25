"""
APD France — API de prédiction
Sert le modèle RandomForest via /predict
"""
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ─── Config ──────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
logger = logging.getLogger("apd-api")

import os
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# ─── Model loading ───────────────────────────────
pipeline = None
meta = None
dropdowns = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, meta, dropdowns
    logger.info("Loading model...")
    pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)
    with open(MODEL_DIR / "dropdowns.json") as f:
        dropdowns = json.load(f)
    logger.info(f"Model loaded — {len(meta['feature_cols'])} features")
    yield


app = FastAPI(
    title="APD France — Prediction API",
    version="1.0.0",
    description="Prédiction du montant d'engagement APD (K EUR) via RandomForest",
    lifespan=lifespan,
)


# ─── Schemas ─────────────────────────────────────
class PredictRequest(BaseModel):
    """Features du projet. Seules les features renseignées sont obligatoires,
    les autres sont imputées par le pipeline sklearn."""
    features: dict = Field(
        ...,
        examples=[{
            "Agence": "Agence française de développement",
            "Type de financement": "Prêt",
            "Pays beneficiaire": "Egypte",
            "Région": "Afrique du Nord",
            "Secteur": "Banques et services financiers",
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


class ModelInfo(BaseModel):
    model_type: str
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
    }


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    """Informations sur le modèle en production."""
    return ModelInfo(
        model_type="RandomForestRegressor",
        n_features=len(meta["feature_cols"]),
        metrics={"rmse_log": meta["rmse"], "mae_log": meta["mae"], "r2_log": meta["r2"]},
        train_size=meta["train_size"],
        test_size=meta["test_size"],
    )


@app.get("/model/features")
def model_features():
    """Liste des features acceptées avec leurs valeurs possibles."""
    return {
        "numerical": meta["num_cols"],
        "categorical_low": meta["cat_low"],
        "categorical_high": meta["cat_high"],
        "dropdowns": dropdowns,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Security(require_api_key)])
def predict(req: PredictRequest):
    """Prédit le montant d'engagement à partir des features du projet."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build input row — fill missing features with None (pipeline imputes)
    row = {}
    for col in meta["feature_cols"]:
        if col in req.features and req.features[col] is not None:
            row[col] = req.features[col]
        elif col in meta["num_cols"]:
            row[col] = 0.0
        else:
            row[col] = None

    # Predict
    X = pd.DataFrame([row])[meta["feature_cols"]]
    log_pred = float(pipeline.predict(X)[0])
    montant_keur = float(np.expm1(log_pred))

    # Classification par quartiles (bornes calculées sur le train complet)
    Q_BOUNDS = [1.792, 2.852, 5.017]
    if log_pred <= Q_BOUNDS[0]:
        tranche = "Petit (≤ 5 K EUR)"
    elif log_pred <= Q_BOUNDS[1]:
        tranche = "Moyen (5 – 16 K EUR)"
    elif log_pred <= Q_BOUNDS[2]:
        tranche = "Grand (16 – 150 K EUR)"
    else:
        tranche = "Très grand (> 150 K EUR)"

    # Fourchette ±1 RMSE
    rmse = meta["rmse"]
    low = float(np.expm1(log_pred - rmse))
    high = float(np.expm1(log_pred + rmse))

    # Label lisible
    if montant_keur < 1000:
        label = f"{montant_keur:,.0f} K EUR"
    else:
        label = f"{montant_keur / 1000:,.1f} M EUR"

    return PredictResponse(
        log_prediction=round(log_pred, 3),
        montant_keur=round(montant_keur, 1),
        montant_label=label,
        tranche=tranche,
        fourchette_keur={"low": round(low, 1), "high": round(high, 1)},
        features_used={k: v for k, v in req.features.items() if v is not None},
    )
