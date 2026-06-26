"""
Prépare les données pour les API :
  1. Copie le CSV brut dans data/raw/
  2. Convertit le CSV propre en parquet dans data/processed/
  3. Vérifie que pipeline.joblib est présent

Usage : python scripts/prepare_data.py <chemin_vers_csv_clean> [<chemin_vers_pipeline.joblib>]
"""
import sys
import shutil
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage : python scripts/prepare_data.py <csv_clean> [<pipeline.joblib>]")
        print("  csv_clean      : aide-publique-au-developpement_clean.csv (séparateur ;)")
        print("  pipeline.joblib: modèle entraîné (optionnel, copié dans api/predict/model/)")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    root = Path(__file__).parent.parent

    # 1. CSV → parquet
    print(f"Conversion {csv_path.name} → parquet...")
    import pandas as pd
    df = pd.read_csv(csv_path, sep=";")
    out_parquet = root / "data" / "processed" / "apd_clean.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"  ✅ {out_parquet} ({len(df)} lignes, {len(df.columns)} colonnes)")

    # 2. Copie du CSV brut
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, raw_dir / csv_path.name)
    print(f"  ✅ {raw_dir / csv_path.name}")

    # 3. pipeline.joblib (si fourni)
    if len(sys.argv) >= 3:
        joblib_path = Path(sys.argv[2])
        dest = root / "api" / "predict" / "model" / "pipeline.joblib"
        shutil.copy2(joblib_path, dest)
        print(f"  ✅ {dest} ({joblib_path.stat().st_size // 1024 // 1024} Mo)")
    else:
        model_path = root / "api" / "predict" / "model" / "pipeline.joblib"
        if not model_path.exists():
            print(f"  ⚠️  {model_path} manquant — l'API predict ne démarrera pas")
            print(f"      Relance avec : python scripts/prepare_data.py {csv_path} /chemin/vers/pipeline.joblib")

    print("\nDonnées prêtes. Lance : docker compose up --build")

if __name__ == "__main__":
    main()
