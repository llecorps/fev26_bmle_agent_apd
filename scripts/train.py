"""
Script de training avec logging MLflow 3.x.

Usage :
  MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/train.py --data data/processed/apd_clean.parquet
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Config ──────────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_NAME = "apd-randomforest"
EXPERIMENT_NAME = "apd-regression"
CHAMPION_ALIAS = "champion"
CARD_THRESHOLD = 15


def load_and_prepare(data_path: str):
    """Charge le dataset et classifie les colonnes."""
    print(f"Chargement de {data_path}...")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, sep=";")
    print(f"  {df.shape[0]} lignes × {df.shape[1]} colonnes")

    TARGET = "log_engagements"
    EXCLUDE = ["Engagements (K EUR)", TARGET]

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in EXCLUDE]

    MARKERS = [c for c in df.columns if c in [
        "Genre", "Aide a l'environnement", "Gouvernance", "Biodiversite",
        "Attenuation du changement climatique", "Adaptation au changement climatique",
        "Desertification", "Développement du commerce",
        "Santé genesique, maternelle, neonatale et infantile (SGMNI)"
    ]]
    for m in MARKERS:
        df[m] = df[m].astype(str)
        if m not in cat_cols:
            cat_cols.append(m)
        if m in num_cols:
            num_cols.remove(m)

    cat_low = [c for c in cat_cols if df[c].nunique() < CARD_THRESHOLD]
    cat_high = [c for c in cat_cols if df[c].nunique() >= CARD_THRESHOLD]
    feature_cols = num_cols + cat_low + cat_high

    return df[feature_cols], df[TARGET], num_cols, cat_low, cat_high, feature_cols


def build_pipeline(num_cols, cat_low, cat_high, **rf_params):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat_low", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_low),
        ("cat_high", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("te", TargetEncoder(smooth="auto", cv=5)),
        ]), cat_high),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)),
    ])


def get_champion_rmse(client, model_name: str) -> tuple[float | None, str | None]:
    """Récupère la RMSE du modèle champion actuel (MLflow 3 aliases)."""
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        run = client.get_run(mv.run_id)
        rmse = float(run.data.metrics.get("test_rmse_log", 999))
        return rmse, mv.version
    except Exception:
        return None, None


def train(data_path: str, n_estimators: int, max_depth: int | None, min_samples_leaf: int):

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow : {MLFLOW_URI}")

    X, y, num_cols, cat_low, cat_high, feature_cols = load_and_prepare(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=pd.qcut(y, q=10, labels=False, duplicates="drop"),
    )
    print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")

    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
    }
    pipeline = build_pipeline(num_cols, cat_low, cat_high, **rf_params)

    run_name = f"rf-{n_estimators}trees-depth{max_depth}-leaf{min_samples_leaf}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'='*60}")
        print(f"Run : {run_name}")
        print(f"{'='*60}")

        # Log params
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "min_samples_leaf": min_samples_leaf,
            "encoding_low": "OneHotEncoder",
            "encoding_high": "TargetEncoder_cv5",
            "n_features": len(feature_cols),
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

        # Cross-validation
        print("\nValidation croisée 5-fold...")
        cv = cross_validate(
            pipeline, X_train, y_train, cv=5,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        )
        cv_rmse = -cv["test_neg_root_mean_squared_error"].mean()
        cv_rmse_std = cv["test_neg_root_mean_squared_error"].std()
        cv_mae = -cv["test_neg_mean_absolute_error"].mean()
        cv_r2 = cv["test_r2"].mean()

        mlflow.log_metrics({
            "cv_rmse_log": round(cv_rmse, 4),
            "cv_rmse_std": round(cv_rmse_std, 4),
            "cv_mae_log": round(cv_mae, 4),
            "cv_r2_log": round(cv_r2, 4),
        })
        print(f"  CV RMSE : {cv_rmse:.4f} ± {cv_rmse_std:.4f}")
        print(f"  CV R²   : {cv_r2:.4f}")

        # Train final
        print("\nEntraînement final...")
        pipeline.fit(X_train, y_train)

        # Evaluate test
        y_pred = pipeline.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metrics({
            "test_rmse_log": round(test_rmse, 4),
            "test_mae_log": round(test_mae, 4),
            "test_r2_log": round(test_r2, 4),
        })
        print(f"\n  Test RMSE : {test_rmse:.4f}")
        print(f"  Test MAE  : {test_mae:.4f}")
        print(f"  Test R²   : {test_r2:.4f}")

        # Log meta.json
        meta = {
            "num_cols": num_cols, "cat_low": cat_low, "cat_high": cat_high,
            "feature_cols": feature_cols,
            "rmse": round(test_rmse, 3), "mae": round(test_mae, 3), "r2": round(test_r2, 3),
            "train_size": len(X_train), "test_size": len(X_test),
        }
        meta_path = "/tmp/meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(meta_path)

        # Log dropdowns.json
        df_full = pd.concat([X_train, X_test])
        dropdowns = {}
        for c in cat_low + cat_high:
            vals = sorted([str(v) for v in df_full[c].dropna().unique().tolist()])
            dropdowns[c] = vals[:200]
        drop_path = "/tmp/dropdowns.json"
        with open(drop_path, "w") as f:
            json.dump(dropdowns, f, ensure_ascii=False, indent=1)
        mlflow.log_artifact(drop_path)

        # Log sklearn model -> registry
        mv = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
            #skops_trusted_types=["numpy.dtype"] # <-- skops doit truster afin d'éviter les erreurs de compatibilité avec les type Numpy
        )

        print(f"\n  Modèle enregistré : {MODEL_NAME}")

        # ─── Promotion via alias "champion" (MLflow 3) ──────
        client = mlflow.tracking.MlflowClient()

        # Récupérer la version qu'on vient d'enregistrer
        latest = client.search_model_versions(
            f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
        )
        new_version = latest[0].version if latest else "1"

        champion_rmse, champion_version = get_champion_rmse(client, MODEL_NAME)

        if champion_rmse is None:
            # Aucun champion → promouvoir directement
            client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
            print(f"  ✅ v{new_version} → alias '{CHAMPION_ALIAS}' (premier modèle)")

        elif test_rmse < champion_rmse * 0.995:
            # Meilleur de > 0.5% → remplacer le champion
            client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
            print(f"  ✅ v{new_version} → alias '{CHAMPION_ALIAS}' "
                  f"(RMSE {test_rmse:.4f} < {champion_rmse:.4f})")

        else:
            # Pas assez meilleur → garder le champion actuel
            client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
            print(f"  ⏸️  v{new_version} → alias 'challenger' "
                  f"(RMSE {test_rmse:.4f} vs champion {champion_rmse:.4f})")

    print(f"\n{'='*60}")
    print(f"UI MLflow : {MLFLOW_URI}")
    print(f"{'='*60}")


if __name__ == "__main__":

    print("\n=== TRAINING RANDOM FOREST ===\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/apd_clean.parquet")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    train(args.data, args.n_estimators, args.max_depth, args.min_samples_leaf)
