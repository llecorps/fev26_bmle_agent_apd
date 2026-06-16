"""Stage `features` : dataset nettoyé -> X/y train/test prêts à l'entraînement.

- Filtre les lignes où la cible (ratio_don par défaut) est absente.
- Encode les colonnes catégorielles en one-hot (drop_first pour éviter la colinéarité).
- Conserve les colonnes numériques et les marqueurs binaires déjà 0/1.
- Sépare train/test avec stratification sur 'type_fin_agrege' si disponible.
- Sauvegarde X_train, X_test, y_train, y_test en parquet dans models/data/.
"""

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "processed" / "apd_clean.parquet"
OUTDIR = ROOT / "models" / "data"
METRICS = ROOT / "metrics" / "features.json"


def main() -> None:
    params_all = yaml.safe_load((ROOT / "params.yaml").read_text())
    params = params_all["features"]
    markers = params_all["prepare"]["markers"]

    df = pd.read_parquet(INPUT)
    target = params["target"]

    df = df.dropna(subset=[target]).reset_index(drop=True)

    available_cat = [c for c in params["categorical"] if c in df.columns]
    available_num = [c for c in params["numerical"] if c in df.columns]
    available_markers = [c for c in markers if c in df.columns]

    feature_frame = pd.concat(
        [
            pd.get_dummies(df[available_cat], drop_first=True, dummy_na=False),
            df[available_num].fillna(df[available_num].median(numeric_only=True)),
            df[available_markers].astype(int),
        ],
        axis=1,
    )
    feature_frame.columns = feature_frame.columns.astype(str)

    y = df[target].astype(float)
    stratify = df["type_fin_agrege"] if "type_fin_agrege" in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=stratify,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(OUTDIR / "X_train.parquet", index=False)
    X_test.to_parquet(OUTDIR / "X_test.parquet", index=False)
    y_train.to_frame(target).to_parquet(OUTDIR / "y_train.parquet", index=False)
    y_test.to_frame(target).to_parquet(OUTDIR / "y_test.parquet", index=False)

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({
        "n_rows_after_dropna": int(len(df)),
        "n_features": int(feature_frame.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_mean_train": float(y_train.mean()),
        "target_mean_test": float(y_test.mean()),
    }, indent=2))
    print(f"features: {feature_frame.shape[1]} features, train={len(X_train):,}, test={len(X_test):,}")


if __name__ == "__main__":
    main()
