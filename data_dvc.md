# Workflow DVC + DagsHub — Journal d'implémentation

Pas-à-pas pour reproduire ce dépôt depuis zéro : pipeline DVC de préparation
des données APD, remote DagsHub, miroir avec GitHub.

---

## 0. Prérequis

- Python 3.10+ et `git` installés.
- Compte **GitHub** (ici : `llecorps`) — sert d'hébergeur du code.
- Compte **DagsHub** (ici : `llecorps`) — sert d'hébergeur du remote DVC + UI MLOps.
- CSV brut OCDE/CAD `aide-publique-au-developpement.csv` (~104 Mo) disponible
  localement.

---

## 1. Arborescence du projet

```text
fev26_bmle_agent_apd/
├── data/
│   ├── processed/      # apd_clean.parquet (sortie de prepare)
│   └── raw/            # CSV source (tracké DVC)
├── metrics/            # features.json (KPI du pipeline, versionné git)
├── models/
│   ├── data/           # X_train, X_test, y_train, y_test en parquet
│   └── models/         # artefacts modèles (vide pour l'instant)
└── src/                # prepare.py, features.py
```

```bash
mkdir -p data/raw data/processed metrics models/data models/models src
```

---

## 2. Environnement Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Créer `requirements.txt` avec :

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
pyarrow>=16.0
pyyaml>=6.0
dvc>=3.55
dvc-s3>=3.2
dagshub>=0.4
```

Installer :

```bash
pip install -r requirements.txt
```

> Note : `dvc-s3` est indispensable, le remote DagsHub utilise l'API S3.

---

## 3. Initialiser DVC

```bash
dvc init
dvc config core.autostage true   # ajoute auto à git les .dvc / .gitignore générés
```

Cela crée `.dvc/` (config + cache) et `.dvcignore`. Le `git status` les place
en staged.

`.gitignore` racine — important : ne PAS ignorer les dossiers entiers,
DVC génère des `.gitignore` locaux par dossier d'output.

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.cache/
```

---

## 4. Tracker le CSV brut avec DVC

```bash
cp /chemin/vers/aide-publique-au-developpement.csv data/raw/
dvc add data/raw/aide-publique-au-developpement.csv
```

Cela crée :
- `data/raw/aide-publique-au-developpement.csv.dvc` (versionné git, contient le hash MD5)
- `data/raw/.gitignore` (ignore le CSV réel pour git)
- Une entrée dans le cache DVC (`.dvc/cache/files/md5/...`)

---

## 5. Paramètres du pipeline

Créer `params.yaml` (hyperparamètres centralisés, voir le fichier dans le repo
pour le contenu complet) :

```yaml
prepare:
  raw_path: data/raw/aide-publique-au-developpement.csv
  csv_separator: ";"
  encoding: utf-8-sig
  markers: [...]

features:
  target: ratio_don
  categorical: [...]
  numerical: [Annee de declaration, mois_engagement]
  test_size: 0.2
  random_state: 42
```

Modifier ce fichier suffit à invalider et rejouer les stages concernées.

---

## 6. Scripts des stages

- `src/prepare.py` — lit le CSV brut, nettoie (drop colonnes USD/codes,
  corrections orthographiques, binarisation marqueurs CAD, calcul du
  `ratio_don`) et écrit `data/processed/apd_clean.parquet`.
- `src/features.py` — encode les catégorielles en one-hot, sépare train/test
  stratifié, écrit `models/data/X_train.parquet`, `X_test.parquet`,
  `y_train.parquet`, `y_test.parquet` et un récap dans `metrics/features.json`.

---

## 7. Pipeline DVC

Créer `dvc.yaml` :

```yaml
stages:
  prepare:
    cmd: python -m src.prepare
    deps:
      - src/prepare.py
      - data/raw/aide-publique-au-developpement.csv
    params: [prepare]
    outs:
      - data/processed/apd_clean.parquet

  features:
    cmd: python -m src.features
    deps:
      - src/features.py
      - data/processed/apd_clean.parquet
    params: [features, prepare.markers]
    outs:
      - models/data/X_train.parquet
      - models/data/X_test.parquet
      - models/data/y_train.parquet
      - models/data/y_test.parquet
    metrics:
      - metrics/features.json:
          cache: false
```

Exécuter le pipeline :

```bash
dvc repro
```

Sortie attendue :

```
Running stage 'prepare':
prepare: 106,519 lignes x 80 colonnes -> data/processed/apd_clean.parquet
Running stage 'features':
features: 117 features, train=58,594, test=14,649
```

Vérifications :

```bash
dvc dag                  # raw -> prepare -> features
dvc status               # "Data and pipelines are up to date."
dvc metrics show         # contenu de metrics/features.json
```

---

## 8. Repo GitHub

Créer le dépôt `llecorps/fev26_bmle_agent_apd` sur GitHub (UI ou
`gh repo create llecorps/fev26_bmle_agent_apd --public`), puis :

```bash
git remote set-url origin https://github.com/llecorps/fev26_bmle_agent_apd.git
git add -A
git commit -m "feat(dvc): pipeline prepare + features"
git push -u origin data
```

---

## 9. Connecter le repo GitHub à DagsHub (mirror)

1. Sur https://dagshub.com → **+ Create → Connect a Repository → GitHub**.
2. Autoriser l'app DagsHub sur le compte GitHub (ici `llecorps`).
3. Sélectionner `llecorps/fev26_bmle_agent_apd`.
4. Mode : **Mirror** (DagsHub se synchronise depuis GitHub).

Le dépôt vit ensuite à
`https://dagshub.com/llecorps/fev26_bmle_agent_apd`.

> Si l'app DagsHub ne voit pas ton repo : vérifie que tu l'as installée sur
> le bon compte GitHub (les apps OAuth sont par utilisateur, pas
> transversales).

---

## 10. Configurer le remote DVC vers DagsHub

DagsHub expose un endpoint S3-compatible par dépôt. Générer un token sur
DagsHub : **avatar → Your Settings → Tokens → New Token** (scope read-write).

```bash
dvc remote add -d origin s3://dvc
dvc remote modify origin  endpointurl https://dagshub.com/llecorps/fev26_bmle_agent_apd.s3
dvc remote modify origin --local access_key_id     <DAGSHUB_TOKEN>
dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
```

- Le token sert à la fois comme `access_key_id` et comme
  `secret_access_key` (volontaire côté DagsHub).
- `--local` met les credentials dans `.dvc/config.local`, **git-ignoré
  par défaut**. Ne JAMAIS commiter ce fichier.

Vérifier :

```bash
dvc remote list      # origin  s3://dvc  (default)
cat .dvc/config      # endpoint visible, pas de token
```

---

## 11. Pousser les données vers DagsHub

```bash
dvc push
# Pushing
# 6 files pushed
```

Les 6 fichiers : `aide-publique-au-developpement.csv` (raw) +
`apd_clean.parquet` + `X_train.parquet` + `X_test.parquet` +
`y_train.parquet` + `y_test.parquet`.

---

## 12. Pousser le code et la conf DVC sur GitHub

```bash
git add .dvc/config .dvcignore .gitignore \
        data/raw/.gitignore data/raw/aide-publique-au-developpement.csv.dvc \
        data/processed/.gitignore models/data/.gitignore \
        dvc.yaml dvc.lock params.yaml requirements.txt README.md \
        src/ metrics/
git status                # contrôler qu'aucun gros binaire n'est staged
git commit -m "chore(dvc): configure DagsHub remote"
git push origin data
```

Le mirror DagsHub se synchronise automatiquement (quelques secondes).
Le commit apparaît sur la page DagsHub, et l'onglet **DVC** liste les
fichiers `.dvc` avec leur taille et un bouton **Download**.

---

## 13. Ajouter `licence.pedago` en lecture seule

Sur la page du dépôt DagsHub :

**Settings → Collaborators → Add Collaborator → `licence.pedago`
→ Role: Reader → Add.**

Le correcteur a alors un accès en lecture au code (via le mirror) et aux
données stockées sur le remote DVC.

---

## 14. Validation de bout en bout (optionnel mais recommandé)

Simuler la reproduction côté correcteur, depuis un dossier propre :

```bash
cd /tmp
git clone https://github.com/llecorps/fev26_bmle_agent_apd.git test-clone
cd test-clone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# credentials nécessaires pour télécharger les données :
dvc remote modify origin --local access_key_id     <DAGSHUB_TOKEN>
dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>

dvc pull        # télécharge data/raw, data/processed, models/data depuis DagsHub
dvc repro       # doit afficher "Data and pipelines are up to date."
```

Si `dvc repro` ne rejoue aucune stage, le lock matche les hashes du remote :
le livrable est 100 % reproductible.

---

## Cheat sheet

| Commande                       | Effet                                                          |
| ------------------------------ | -------------------------------------------------------------- |
| `dvc repro`                    | Rejoue les stages dont les inputs ont changé                   |
| `dvc dag`                      | Affiche le graphe des stages                                   |
| `dvc status`                   | Vérifie cohérence entre code, données et lock                  |
| `dvc metrics show`             | Affiche le contenu des fichiers métriques                      |
| `dvc params diff`              | Diff des params.yaml entre commits                             |
| `dvc push` / `dvc pull`        | Synchronise les données avec le remote DagsHub                 |
| `dvc add <file>`               | Tracke un fichier hors d'un stage de pipeline                  |
| `dvc remote list`              | Liste les remotes configurés                                   |
