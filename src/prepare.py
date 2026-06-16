"""Stage `prepare` : CSV brut APD -> dataset nettoyé en parquet.

Reproduit les transformations principales de 02_PreProcessing_FeatureEng.ipynb :
- supprime les colonnes redondantes (versions USD, codes, équivalents),
- normalise les dates en mois_engagement,
- corrige les variantes orthographiques (Türkiye, FMI, BM, ...),
- binarise les marqueurs CAD (Genre, Environnement, ...),
- calcule la cible ratio_don = Équivalent don / Engagements (clippée [0, 1]).
"""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "processed" / "apd_clean.parquet"

CORRECTIONS = {
    "Pays beneficiaire": {"Türkiye": "Turquie"},
    "Canal agrege": {
        "Agence, fonds ou commission des Nations unies (NU)": "Agence, fonds ou commission des Nations Unies (NU)",
        "Fonds monétaire international (FMI)": "Fonds Monétaire International (FMI)",
        "Groupe de la Banque mondiale (BM)": "Groupe de la Banque Mondiale (BM)",
    },
}


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["prepare"]

    df = pd.read_csv(ROOT / params["raw_path"], sep=params["csv_separator"], low_memory=False)

    # Colonnes redondantes : versions USD, codes, descriptions libres.
    df = df.loc[:, ~df.columns.str.contains("USD", regex=False)]
    df = df.loc[:, ~df.columns.str.contains("code|Code|Codes|codes")]

    # Date d'engagement -> mois (l'année est redondante avec 'Annee de declaration').
    df["Annee de declaration"] = df["Annee de declaration"].astype(int)
    df["Date d'engagement"] = pd.to_datetime(df["Date d'engagement"], errors="coerce")
    df["mois_engagement"] = df["Date d'engagement"].dt.month
    df = df.drop(columns=["Date d'engagement"])

    # Corrections orthographiques.
    for col, mapping in CORRECTIONS.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # Marqueurs CAD : NaN -> 0, valeur positive -> 1.
    for col in params["markers"]:
        if col in df.columns:
            df[col] = (df[col].fillna(0) > 0).astype(int)

    # Cible : ratio_don. Équivalent don / Engagements, dans [0, 1].
    eng = pd.to_numeric(df.get("Engagements (K EUR)"), errors="coerce")
    don = pd.to_numeric(df.get("Equivalent don (K EUR)"), errors="coerce")
    df["ratio_don"] = (don / eng).where(eng > 0).clip(lower=0, upper=1)

    # type_fin_agrege : Don / Prêt / Autre (utile pour stratification et EDA).
    type_fin = df.get("Type de financement")
    if type_fin is not None:
        df["type_fin_agrege"] = type_fin.map(
            lambda v: "Don" if v == "Dons" else ("Prêt" if v == "Prêt" else "Autre")
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"prepare: {len(df):,} lignes x {df.shape[1]} colonnes -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
