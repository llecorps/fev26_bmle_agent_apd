"""Prépare les jeux de données de MONITORING pour l'endpoint /evaluate.

À partir du CSV brut APD, on découpe par mois de `Date d'engagement` :
  - référence  : projets engagés en JANVIER  -> reference_january.parquet
  - courant    : projets engagés en FÉVRIER  -> current_february.parquet

Chaque parquet contient les colonnes de features du modèle + la cible réelle
`Engagements (K EUR)`, ce qui permet à /evaluate de :
  1. prédire avec le modèle,
  2. comparer prédictions vs cible (RMSE/MAE/R²),
  3. mesurer la dérive des features (Evidently) entre référence et courant.

Usage : python scripts/prepare_monitoring.py
Sortie : data/monitoring/{reference_january,current_february}.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "monitoring"

DATE_COL = "Date d'engagement"
TARGET = "Engagements (K EUR)"

# Colonnes attendues par le modèle (cf. models/data/meta.json -> feature_cols).
FEATURE_COLS = [
    "Agence", "Type de financement", "Pays beneficiaire", "Région",
    "Secteur", "Catégorie CAD", "Bi/Multi", "Annee de declaration",
]


def _clean_amount(series: pd.Series) -> pd.Series:
    """Montant format français (virgule décimale, espace milliers) -> float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["prepare"]
    df = pd.read_csv(
        ROOT / params["raw_path"],
        sep=params["csv_separator"],
        encoding=params.get("encoding", "utf-8-sig"),
        low_memory=False,
    )
    print(f"CSV brut : {len(df):,} lignes")

    # Dates au format ISO (YYYY-MM-DD) dans le CSV brut ; "mixed" tolère les variantes.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format="mixed")
    df[TARGET] = _clean_amount(df[TARGET])

    # On garde features + cible + mois, et on ne conserve que les engagements > 0
    # (cible = log1p(engagement), cohérent avec l'entraînement du modèle).
    keep = [c for c in FEATURE_COLS if c in df.columns] + [TARGET, DATE_COL]
    df = df[keep].dropna(subset=[TARGET, DATE_COL])
    df = df[df[TARGET] > 0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for month, name in [(1, "reference_january"), (2, "current_february")]:
        sub = df[df[DATE_COL].dt.month == month].drop(columns=[DATE_COL]).copy()
        out = OUT_DIR / f"{name}.parquet"
        sub.to_parquet(out, index=False)
        print(f"  ✅ {out.name} : {len(sub):,} lignes")


if __name__ == "__main__":
    main()
