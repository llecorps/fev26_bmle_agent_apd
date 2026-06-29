"""
DAG APD — Pipeline horaire de mise à jour des données et du modèle.

Étapes :
  1  → télécharge le CSV depuis DagsHub S3
  2  → nettoie et filtre les données
  3  → feature engineering
  4' → entraîne LinearRegression      (XCom : score CV)
  4'' → entraîne DecisionTreeRegressor (XCom : score CV)
  4''' → entraîne RandomForestRegressor (XCom : score CV)
  5  → sélectionne le meilleur modèle, réentraîne sur tout, sauvegarde
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# ── Chemins montés dans les conteneurs ───────────────────────────────────────
RAW_DIR   = "/app/raw_files"
CLEAN_DIR = "/app/clean_data"
MODEL_DIR = "/app/models/data"

# ── Config DagsHub S3 ─────────────────────────────────────────────────────────
# DVC stocke les fichiers par hash MD5 (files/md5/<2 premiers>/<reste>), pas par
# nom logique. Le MD5 vient de data/raw/aide-publique-au-developpement.csv.dvc.
DAGSHUB_ENDPOINT = "https://dagshub.com/llecorps/fev26_bmle_agent_apd.s3"
DAGSHUB_BUCKET   = "dvc"
RAW_CSV_MD5      = "87657ce5b6f2da554db6c25b4450dabd"
DAGSHUB_KEY      = f"files/md5/{RAW_CSV_MD5[:2]}/{RAW_CSV_MD5[2:]}"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 1 — Téléchargement depuis DagsHub S3
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    import boto3
    from botocore.client import Config
    from datetime import datetime

    access_key = os.environ["DAGSHUB_ACCESS_KEY"]
    secret_key = os.environ["DAGSHUB_SECRET_KEY"]

    s3 = boto3.client(
        "s3",
        endpoint_url=DAGSHUB_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    filename = f"apd_{datetime.now().strftime('%Y-%m-%d_%H')}.csv"
    dest = os.path.join(RAW_DIR, filename)

    print(f"Téléchargement → {dest}")
    s3.download_file(DAGSHUB_BUCKET, DAGSHUB_KEY, dest)
    print(f"Fichier téléchargé : {os.path.getsize(dest) / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 2 — Nettoyage des données brutes
# ─────────────────────────────────────────────────────────────────────────────
def clean_data():
    import pandas as pd
    import glob

    files = sorted(glob.glob(os.path.join(RAW_DIR, "apd_*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV dans {RAW_DIR}")

    latest = files[-1]
    print(f"Lecture de {latest}")
    df = pd.read_csv(latest, sep=";", low_memory=False)

    # Nettoyage de base
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Montant verse (K EUR)", "Pays beneficiaire", "Secteur"])
    df = df[df["Montant verse (K EUR)"] > 0]
    df["Annee de declaration"] = pd.to_numeric(df["Annee de declaration"], errors="coerce")
    df = df.dropna(subset=["Annee de declaration"])
    df["Annee de declaration"] = df["Annee de declaration"].astype(int)

    os.makedirs(CLEAN_DIR, exist_ok=True)
    out = os.path.join(CLEAN_DIR, "apd_clean.csv")
    df.to_csv(out, index=False)
    print(f"Données nettoyées : {len(df)} lignes → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 3 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineering():
    import pandas as pd
    import numpy as np

    src = os.path.join(CLEAN_DIR, "apd_clean.csv")
    df = pd.read_csv(src, low_memory=False)

    # Variable cible : log(1 + montant versé)
    df["log_montant"] = np.log1p(df["Montant verse (K EUR)"])

    # Features catégorielles retenues
    CAT_COLS = [
        "Agence", "Type de financement", "Pays beneficiaire",
        "Région", "Secteur", "Catégorie CAD", "Bi/Multi",
    ]
    NUM_COLS = ["Annee de declaration"]

    keep = CAT_COLS + NUM_COLS + ["log_montant"]
    df = df[[c for c in keep if c in df.columns]].dropna()

    out = os.path.join(CLEAN_DIR, "apd_features.csv")
    df.to_csv(out, index=False)
    print(f"Features engineering : {len(df)} lignes, {df.shape[1]} colonnes → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHES 4', 4'', 4''' — Entraînement + score CV (XCom)
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(model_name: str, **context):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    import numpy as np

    df = pd.read_csv(os.path.join(CLEAN_DIR, "apd_features.csv"), low_memory=False)

    CAT_COLS = [c for c in df.columns if df[c].dtype == object]
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS and c != "log_montant"]
    X = df[CAT_COLS + NUM_COLS]
    y = df["log_montant"]

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    models = {
        "LinearRegression":       LinearRegression(),
        "DecisionTreeRegressor":  DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForestRegressor":  RandomForestRegressor(n_estimators=100, max_depth=10,
                                                        n_jobs=-1, random_state=42),
    }

    pipe = Pipeline([("prep", preprocessor), ("model", models[model_name])])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    score = float(np.mean(scores))

    print(f"{model_name} — R² CV moyen : {score:.4f}")
    context["ti"].xcom_push(key=f"score_{model_name}", value=score)
    return score


def train_linear(**ctx):
    _train_model("LinearRegression", **ctx)

def train_tree(**ctx):
    _train_model("DecisionTreeRegressor", **ctx)

def train_forest(**ctx):
    _train_model("RandomForestRegressor", **ctx)


# ─────────────────────────────────────────────────────────────────────────────
# TÂCHE 5 — Sélection du meilleur modèle, réentraînement, sauvegarde
# ─────────────────────────────────────────────────────────────────────────────
def select_and_save(**context):
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    ti = context["ti"]
    scores = {
        "LinearRegression":      ti.xcom_pull(key="score_LinearRegression"),
        "DecisionTreeRegressor": ti.xcom_pull(key="score_DecisionTreeRegressor"),
        "RandomForestRegressor": ti.xcom_pull(key="score_RandomForestRegressor"),
    }
    print(f"Scores CV : {scores}")
    best_name = max(scores, key=scores.get)
    print(f"Meilleur modèle : {best_name} (R²={scores[best_name]:.4f})")

    df = pd.read_csv(os.path.join(CLEAN_DIR, "apd_features.csv"), low_memory=False)
    CAT_COLS = [c for c in df.columns if df[c].dtype == object]
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS and c != "log_montant"]
    X = df[CAT_COLS + NUM_COLS]
    y = df["log_montant"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression":       LinearRegression(),
        "DecisionTreeRegressor":  DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForestRegressor":  RandomForestRegressor(n_estimators=100, max_depth=10,
                                                        n_jobs=-1, random_state=42),
    }

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
    pipe.fit(X, y)

    # Métriques sur le test set
    y_pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODEL_DIR, "pipeline.joblib"))

    meta = {
        "model": best_name,
        "cv_scores": scores,
        "rmse": rmse, "mae": mae, "r2": r2,
        "feature_cols": CAT_COLS + NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Modèle sauvegardé : {best_name} | RMSE={rmse:.4f} R²={r2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Définition du DAG
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="apd_pipeline",
    default_args=default_args,
    description="Pipeline horaire APD : download → clean → features → train → select",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
    tags=["apd", "ml"],
) as dag:

    t1 = PythonOperator(task_id="download_data",       python_callable=download_data)
    t2 = PythonOperator(task_id="clean_data",          python_callable=clean_data)
    t3 = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering)

    t4a = PythonOperator(task_id="train_linear",  python_callable=train_linear,  provide_context=True)
    t4b = PythonOperator(task_id="train_tree",    python_callable=train_tree,    provide_context=True)
    t4c = PythonOperator(task_id="train_forest",  python_callable=train_forest,  provide_context=True)

    t5 = PythonOperator(task_id="select_and_save", python_callable=select_and_save, provide_context=True)

    t1 >> t2 >> t3 >> [t4a, t4b, t4c] >> t5
