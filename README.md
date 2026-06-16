# APD — Workflow de modélisation (DVC + DagsHub)

Pipeline reproductible de préparation des données pour la modélisation du
**ratio de don** (`ratio_don`) sur les déclarations françaises d'Aide Publique
au Développement (CSV brut OCDE/CAD, ~106 000 lignes).

## Pipeline

```
data/raw/aide-publique-au-developpement.csv  (tracké DVC)
              │
              ▼  src/prepare.py
data/processed/apd_clean.parquet
              │
              ▼  src/features.py
models/data/{X_train, X_test, y_train, y_test}.parquet
metrics/features.json
```

Vérification : `dvc dag`.

## Reproduire le pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/dvc pull          # récupère les données depuis le remote DagsHub
.venv/bin/dvc repro         # rejoue les stages dont les inputs ont changé
.venv/bin/dvc metrics show  # affiche metrics/features.json
```

Sans `dvc pull`, fournir manuellement `data/raw/aide-publique-au-developpement.csv`
puis lancer `dvc repro`.

## Modifier le pipeline

- **Hyperparamètres** : éditer `params.yaml` (séparateur CSV, marqueurs CAD,
  colonnes encodées en one-hot, taille du test set, seed). DVC détecte la
  modification et ne réexécute que les stages impactées.
- **Logique** : éditer `src/prepare.py` ou `src/features.py`.
- Après modification : `dvc repro` puis `git add` + `git commit` des fichiers
  versionnés par git (`dvc.lock`, `*.dvc`, code, `params.yaml`,
  `metrics/features.json`).

## Remote DagsHub

DagsHub expose un endpoint S3-compatible pour stocker les données DVC.

```bash
.venv/bin/dvc remote add -d origin s3://dvc
.venv/bin/dvc remote modify origin  endpointurl https://dagshub.com/<user>/<repo>.s3
.venv/bin/dvc remote modify origin --local access_key_id     <DAGSHUB_TOKEN>
.venv/bin/dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
.venv/bin/dvc push          # envoie data/raw, data/processed, models/data, ...
```

Le token se génère depuis l'onglet *Settings → Tokens* du dépôt DagsHub.
Les credentials vont dans `.dvc/config.local` (gitignoré) — ne JAMAIS commiter.

Côté git, le dépôt DagsHub est un miroir GitHub :

```bash
git remote add dagshub https://dagshub.com/<user>/<repo>.git
git push dagshub <branche>
```

## Arborescence

```
data/
  raw/        # CSV source APD (tracké par DVC)
  processed/  # apd_clean.parquet (sortie de prepare)
metrics/      # metrics/features.json (KPI du pipeline, versionné git)
models/
  data/       # X_train, X_test, y_train, y_test en parquet
  models/     # artefacts modèles (vide pour l'instant)
src/
  prepare.py  # stage prepare
  features.py # stage features
dvc.yaml      # définition des stages
params.yaml   # hyperparamètres
dvc.lock      # hashes des inputs/outputs (versionné git, géré par DVC)
```
