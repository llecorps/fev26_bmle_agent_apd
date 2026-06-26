"""
DAG dédié — prépare uniquement aide-publique-au-developpement_clean.csv.

Reprend la logique de src/prepare_data.py :
  1. download_clean_csv : récupère le CSV propre depuis DagsHub (DVC remote S3)
  2. csv_to_parquet     : convertit le CSV (séparateur ;) en parquet
                          dans data/processed/apd_clean.parquet (lu par les API)

Planifié toutes les heures pour garder le parquet à jour.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import shutil

# ── Chemins montés dans le conteneur ─────────────────────────────────────────
RAW_DIR       = "/app/raw_files"
PROCESSED_DIR = "/app/data/processed"
CSV_NAME      = "aide-publique-au-developpement_clean.csv"
PARQUET_NAME  = "apd_clean.parquet"

# ── Config DagsHub S3 (DVC remote) ───────────────────────────────────────────
# DVC stocke les fichiers par hash MD5 : files/md5/<2 premiers>/<reste>.
DAGSHUB_ENDPOINT = "https://dagshub.com/llecorps/fev26_bmle_agent_apd.s3"
DAGSHUB_BUCKET   = "dvc"
CLEAN_CSV_MD5    = "e41e1c6cb177c6168dc0d23a0e6526e6"
DAGSHUB_KEY      = f"files/md5/{CLEAN_CSV_MD5[:2]}/{CLEAN_CSV_MD5[2:]}"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def download_clean_csv():
    """Télécharge le CSV propre depuis le remote DVC de DagsHub."""
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=DAGSHUB_ENDPOINT,
        aws_access_key_id=os.environ["DAGSHUB_ACCESS_KEY"],
        aws_secret_access_key=os.environ["DAGSHUB_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, CSV_NAME)
    print(f"Téléchargement s3://{DAGSHUB_BUCKET}/{DAGSHUB_KEY} → {dest}")
    s3.download_file(DAGSHUB_BUCKET, DAGSHUB_KEY, dest)
    print(f"  ✅ {os.path.getsize(dest) / 1e6:.1f} MB")


def csv_to_parquet():
    """Convertit le CSV propre (séparateur ;) en parquet pour les API."""
    import pandas as pd

    src = os.path.join(RAW_DIR, CSV_NAME)
    if not os.path.exists(src):
        raise FileNotFoundError(f"CSV introuvable : {src}")

    print(f"Lecture de {src} (séparateur ';')...")
    df = pd.read_csv(src, sep=";", low_memory=False)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, PARQUET_NAME)
    df.to_parquet(out, index=False)
    print(f"  ✅ {out} ({len(df)} lignes, {len(df.columns)} colonnes)")


with DAG(
    dag_id="apd_prepare_clean",
    default_args=default_args,
    description="Prépare aide-publique-au-developpement_clean.csv → parquet",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
    tags=["apd", "data"],
) as dag:

    t_download = PythonOperator(
        task_id="download_clean_csv",
        python_callable=download_clean_csv,
    )

    t_convert = PythonOperator(
        task_id="csv_to_parquet",
        python_callable=csv_to_parquet,
    )

    t_download >> t_convert
