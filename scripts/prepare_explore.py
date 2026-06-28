"""Stage `prepare_explore` : CSV brut APD -> dataset d'EXPLORATION en parquet.

Destiné au chatbot d'exploration (api/explore), PAS au modèle de prédiction.

Différence avec `src/prepare.py` (dataset de modélisation) : on applique le même
NETTOYAGE que le notebook d'analyse (normalisation des types au format français,
fusions de libellés, déduplication par projet SNPC, élagage des colonnes vides et
redondantes), mais on N'APPLIQUE PAS les réductions spécifiques au modèle :

  - on CONSERVE la dimension temporelle, en PRIORISANT la date d'engagement
    (date de signature du contrat) : 'date_engagement' (datetime),
    'annee_engagement' et 'mois_engagement'. L'année de déclaration (simple
    année de saisie système) est écartée ;
  - on CONSERVE les libellés lisibles (ODD en texte, secteurs, pays, canaux…) au
    lieu de les binariser en colonnes ODD_* ;
  - on CONSERVE les montants en K EUR (le « leakage » ne concerne que le modèle ;
    pour l'exploration, « montant versé par pays » est une question légitime).

Version « allégée » : normalisation des types + dédup SNPC + fusions principales
de libellés + colonnes temporelles conservées. On omet volontairement la
machinerie propre au modèle (multi-hot ODD, imputation de la cible, corrections
de barèmes des marqueurs).

Sortie : data/processed/apd_explore.parquet
Exécution : python -m src.prepare_explore   (ou via `dvc repro prepare_explore`)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "processed" / "apd_explore.parquet"

# Clé de projet servant à la déduplication (cf. notebook d'analyse, étape doublons).
SNPC_KEY = "Num identification SNPC"
DATE_ENGAGEMENT = "Date d'engagement"
ENGAGEMENTS = "Engagements (K EUR)"

# Colonnes redondantes / non exploitables à retirer (libellé conservé ailleurs).
CODES_REDONDANTS = [
    "Code agence", "Code nature de l'activite", "Code du pays beneficiaire",
    "Code canal de transfert", "Code canal parent", "Bi/Multi",
    "Code type de flux", "Code type de financement",
    "Code modalite de cooperation", "Code objet",
    "Code aide regionale aux PMA", "Code outil de mobilisation des flux prives",
    "Code origine des fonds mobilises", "Code profil de remboursement",
    "Code Marqueur ISP", "code secteur CAD",
]
IDENTIFIANTS = ["Numero de projet", "Codes ISO"]
DESCRIPTIONS_VERBEUSES = [
    "Description courte", "Description", "Mots cles",
    "Description de l'additionnalite", "Additionnalite - objectif de développement",
]
# Dates non pertinentes pour l'exploration (très lacunaires). On garde
# 'Date d'engagement' jusqu'à la déduplication, puis on en dérive le mois.
AUTRES_DATES = [
    "Date prevue de lancement", "Date prevue de fin",
    "Date de premier remboursement",
    "Date finale de remboursement ou echeance attendue pour les actions",
]
DIVERS_INUTILES = ["Localisation geographique", "Type de finance mixte"]

# Fusions de libellés (portage des cas principaux du notebook d'analyse).
FUSIONS_CANAL = {
    "Cités Unies Frances": "Cités Unies France",
    "Gouvernment central": "Gouvernement central",
    "Central Government": "Gouvernement central",
    "Gouvernment local": "Gouvernement local",
    "ONG INTERNATIONALE": "ONG internationale",
    "ONG internationales": "ONG internationale",
    "Other public entities in donor country": "Autre entité publique dans le pays donneur",
    "Institutions du Secteur Privé": "Institutions du secteur privé",
    "Réseaux": "Réseau",
    "RÉSEAU": "Réseau",
    "Agence, fonds ou commission des Nations Unies (ONU)": "Agence, fonds ou commission des Nations unies (NU)",
    "INTERNATIONAL UNION FOR CONSERVATION OF NATURE": "Union internationale pour la conservation de la nature",
    "ACTION AGAINST HUNGER": "Action contre la faim",
    "ACTION CONTRE LA FAIM": "Action contre la faim",
    "ACTION CONTRE LA FAIM Espagne": "Action contre la faim",
    "Fundacion Accion contra el hambre": "Action contre la faim",
}
FUSIONS_CANAL_AGREGE = {
    "Agence, fonds ou commission des Nations Unies (NU)": "Agence, fonds ou commission des Nations unies (NU)",
    "Groupe de la Banque Mondiale (BM)": "Groupe de la Banque mondiale (BM)",
    "Fonds Monétaire International (FMI)": "Fonds monétaire international (FMI)",
}
FUSIONS_AGENCE = {
    "Ministère de l'enseignement supérieur et de la recherche (MESR)":
        "Ministère de l'enseignement supérieur et de la recherche",
    "Ministère de l'économie et des finances":
        "Ministère de l'économie, des finances, et de la souveraineté industrielle",
}
FUSIONS_PAYS = {"Türkiye": "Turquie"}


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["prepare"]

    df = pd.read_csv(
        ROOT / params["raw_path"],
        sep=params["csv_separator"],
        encoding=params.get("encoding", "utf-8-sig"),
        low_memory=False,
    )
    print(f"[0] CSV brut : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")

    # ── 1. Normalisation des types ──────────────────────────────────────
    # Date d'engagement (format français, jour en premier).
    if DATE_ENGAGEMENT in df.columns:
        df[DATE_ENGAGEMENT] = pd.to_datetime(df[DATE_ENGAGEMENT], errors="coerce", dayfirst=True)

    # Montants : virgule décimale + espace milliers -> float.
    montant_cols = [c for c in df.columns
                    if any(kw in c for kw in ["K EUR", "K USD", "M EUR", "M USD"])]
    for col in montant_cols:
        df[col] = (df[col].astype(str)
                   .str.replace(",", ".", regex=False)
                   .str.replace(" ", "", regex=False)
                   .replace("nan", np.nan))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Année de déclaration : écartée plus loin (simple année de saisie système),
    # on ne perd pas de temps à la typer ici.

    # Texte : apostrophe courbe -> droite, espaces parasites.
    for col in df.select_dtypes(include="object").columns:
        df[col] = (df[col].astype(str)
                   .str.replace("\u2019", "'", regex=False)
                   .str.strip()
                   .str.replace(r"\s+", " ", regex=True)
                   .replace("nan", pd.NA))
    print("[1] Types normalisés (dates, montants, année, texte)")

    # ── 2. Suppression des colonnes redondantes / inutiles ──────────────
    # (on NE retire PAS les montants K EUR : utiles pour l'exploration)
    df = df.loc[:, ~df.columns.str.contains("(K USD)", regex=False)]
    for c in ["Montant versé (M EUR)", "Équivalent don (M EUR)"]:
        if c in df.columns:
            df = df.drop(columns=c)
    to_drop = (CODES_REDONDANTS + IDENTIFIANTS + DESCRIPTIONS_VERBEUSES
               + AUTRES_DATES + DIVERS_INUTILES)
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    print(f"[2] Colonnes redondantes retirées -> {df.shape[1]} colonnes")

    # ── 3. Élagage des colonnes trop vides (> 90 % NA) ──────────────────
    # On protège les colonnes structurantes pour ne jamais les perdre.
    protege = {ENGAGEMENTS, DATE_ENGAGEMENT, SNPC_KEY}
    missing_pct = df.isna().mean() * 100
    creuses = [c for c in missing_pct[missing_pct > 90].index if c not in protege]
    if creuses:
        df = df.drop(columns=creuses)
    print(f"[3] {len(creuses)} colonnes creuses (>90% NA) retirées -> {df.shape[1]} colonnes")

    # ── 4. Fusions de libellés (groupby fiables) ────────────────────────
    if "Canal de transfert" in df.columns:
        df["Canal de transfert"] = df["Canal de transfert"].replace(FUSIONS_CANAL)
    if "Canal agrege" in df.columns:
        df["Canal agrege"] = df["Canal agrege"].replace(FUSIONS_CANAL_AGREGE)
    if "Agence" in df.columns:
        df["Agence"] = df["Agence"].replace(FUSIONS_AGENCE)
    if "Pays beneficiaire" in df.columns:
        df["Pays beneficiaire"] = df["Pays beneficiaire"].replace(FUSIONS_PAYS)
    if "Origine des fonds mobilises" in df.columns:
        df["Origine des fonds mobilises"] = df["Origine des fonds mobilises"].str.capitalize()
    print("[4] Fusions de libellés appliquées")

    # ── 5. Marqueurs CAD -> binaire (0/1) ───────────────────────────────
    for col in params.get("markers", []):
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).astype(int)

    # ── 6. Déduplication au niveau projet (SNPC) ────────────────────────
    # Sans cette étape, un même projet déclaré en plusieurs lignes gonfle les
    # totaux. On somme les montants K EUR par projet puis on garde une ligne.
    if SNPC_KEY in df.columns:
        nb0 = len(df)
        subset = [c for c in [DATE_ENGAGEMENT, ENGAGEMENTS, SNPC_KEY] if c in df.columns]
        df = df.drop_duplicates(subset=subset)

        sum_cols = [c for c in df.columns
                    if "K EUR" in c and pd.api.types.is_numeric_dtype(df[c])]
        for c in sum_cols:
            df[c] = df.groupby(SNPC_KEY)[c].transform("sum")

        sort_cols = [c for c in [SNPC_KEY, DATE_ENGAGEMENT] if c in df.columns]
        df = df.sort_values(by=sort_cols)
        df = df.drop_duplicates(subset=[SNPC_KEY], keep="first")
        print(f"[6] Dédup SNPC : {nb0:,} -> {len(df):,} lignes (1 ligne / projet)")

    # ── 7. Dimension temporelle : on PRIORISE la date d'engagement ──────
    # 'Date d'engagement' = date de signature du contrat (info analytique).
    # 'Annee de declaration' = simple année de saisie système -> on l'écarte.
    # On conserve une VRAIE colonne datetime (renommée sans apostrophe pour
    # éviter le piège de quoting), plus l'année et le mois dérivés.
    if DATE_ENGAGEMENT in df.columns:
        d = df[DATE_ENGAGEMENT]
        df["date_engagement"] = d                      # datetime (séries temporelles)
        df["annee_engagement"] = d.dt.year.astype("Int64")   # entier (groupby simple)
        df["mois_engagement"] = d.dt.month.astype("Int64")   # 1-12 (saisonnalité)
        df = df.drop(columns=[DATE_ENGAGEMENT])
    if "Annee de declaration" in df.columns:
        df = df.drop(columns=["Annee de declaration"])
    if SNPC_KEY in df.columns:
        df = df.drop(columns=[SNPC_KEY])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"[FINAL] Dataset d'exploration : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    print(f"        -> {OUTPUT.relative_to(ROOT)}")
    temporelles = [c for c in df.columns
                   if c in ("date_engagement", "annee_engagement", "mois_engagement")]
    print(f"        Colonnes temporelles conservées : {temporelles}")


if __name__ == "__main__":
    main()