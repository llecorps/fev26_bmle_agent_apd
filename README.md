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

## Chatbot analytique (api + ui + llm)

Au-dessus des données préparées, un chatbot répond à des questions en langage
naturel : un LLM génère du code pandas, l'API le valide puis l'exécute, et
renvoie le résultat à l'UI.

```
Streamlit (ui)  ──HTTP──►  FastAPI (api)  ──HTTP──►  serveur LLM
   :8500                      :8080 (interne)          :8000 (hôte)
                                │
                                ▼
                  1. génère le code pandas (LLM)
                  2. validation AST (api/sandbox.py)
                  3. exécution sandboxée (python -I, timeout)
                  4. lit data/processed/apd_clean.parquet (monté :ro)
```

### ⚠️ Pourquoi le LLM n'est PAS dans docker compose

Le serveur LLM (`vllm-mlx`) utilise le **GPU Metal/MPS** d'Apple Silicon
(`VLLM_TARGET_DEVICE=mps`). Or sur macOS, les conteneurs Docker tournent dans
une VM Linux **sans accès au GPU Metal**. `vllm-mlx` ne peut donc pas être
containerisé sur Mac.

Conséquence : le LLM tourne en **process hôte** (lancé via
`llm/start_vllm_mac.sh`), et l'API conteneurisée le joint via
`host.docker.internal:8000`. Seuls `api` et `ui` sont gérés par docker compose.

> Pour un déploiement Linux + GPU NVIDIA, le LLM pourrait être ajouté comme
> service compose (vLLM CUDA). Sur Mac, le garder côté hôte est la seule option
> qui exploite le GPU.

### Démarrer le chatbot

```bash
# 1. (une fois) renseigner le token Hugging Face — optionnel pour un modèle public
cp llm/.env.example llm/.env        # puis éditer HF_TOKEN si nécessaire

# 2. construire les données si besoin
make data            # ou make pull

# 3. terminal A — serveur LLM (process hôte, GPU Metal)
make llm

# 4. terminal B — API + UI dockerisés
make up
```

- UI : http://localhost:8500
- API : http://localhost:8081/explore (port hôte ; interne 8080)

Le port hôte de l'API est **8081** par défaut pour éviter la collision avec
d'autres services sur 8080. Surchargeable : `API_PORT=9000 make up`
(idem `UI_PORT`). Le port interne reste 8080, donc l'UI joint l'API sans
changement via le réseau Docker.

Arrêt : `make down`. Logs : `make logs`.

### Sécurité d'exécution

Le code généré par le LLM est exécuté, donc encadré en profondeur :

1. **Validation AST** ([api/sandbox.py](api/sandbox.py)) avant toute exécution :
   imports limités à une whitelist (pandas, numpy, json…), blocage de
   `eval`/`exec`/`open`, des dunders et des méthodes d'écriture fichier
   (`to_csv`, `to_pickle`…). Testé : `make test` (18 cas).
2. **subprocess isolé** : `python -I`, avec un **timeout** (`EXEC_TIMEOUT`, 30 s).
3. **Volume données en lecture seule** (`./data:/data:ro`) : pas d'écriture ni
   d'exfiltration possible vers les données.

### Variables d'environnement (api)

| Variable       | Défaut                                          | Rôle                          |
| -------------- | ----------------------------------------------- | ----------------------------- |
| `DATA_PATH`    | `/data/processed/apd_clean.parquet`             | Fichier lu par le code généré |
| `LLM_URL`      | `http://host.docker.internal:8000/v1/chat/...`  | Endpoint du serveur LLM       |
| `LLM_MODEL`    | `mlx-community/Mistral-7B-Instruct-v0.3-4bit`   | Modèle servi                  |
| `EXEC_TIMEOUT` | `30`                                            | Timeout d'exécution (s)       |

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
api/          # FastAPI : génération + validation + exécution du code pandas
  explore.py  #   endpoint /explore
  sandbox.py  #   validation AST du code généré
  test_sandbox.py
ui/           # Streamlit : interface de chat (app.py)
llm/          # serveur LLM hôte (start_vllm_mac.sh, .env.example)
docker-compose.yml  # services api + ui
Makefile      # orchestration (make help)
dvc.yaml      # définition des stages
params.yaml   # hyperparamètres
dvc.lock      # hashes des inputs/outputs (versionné git, géré par DVC)
```
