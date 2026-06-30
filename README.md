# APD — Agent analytique & MLOps

Plateforme d'exploration et de prédiction sur les déclarations françaises d'Aide
Publique au Développement (~106 000 lignes) : un **chatbot** qui répond à des
questions en langage naturel, une **API de prédiction** des montants, le tout
orchestré par **Airflow**, tracé par **MLflow** et visualisé via un **dashboard**.

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
                  2. validation AST (api/explore/sandbox.py)
                  3. exécution sandboxée (python -I, timeout)
                  4. lit data/processed/apd_explore.parquet (monté :ro)
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

### Démarrer la stack

```bash
# 1. (une fois) renseigner le token Hugging Face — optionnel pour un modèle public
cp llm/.env.example llm/.env        # puis éditer HF_TOKEN si nécessaire

# 2. (Airflow) credentials DagsHub dans un .env à la racine (gitignoré)
#    DAGSHUB_ACCESS_KEY=...  /  DAGSHUB_SECRET_KEY=...
set -a && source .env && set +a

# 3. construire les données si besoin
make data            # ou make pull

# 4. terminal A — serveur LLM (process hôte, GPU Metal)
make llm

# 5. terminal B — toute la stack dockerisée (api, ui, mlflow, airflow, dashboard)
make up
```

### Services & URLs

| Service        | URL                                   | Détails                              |
| -------------- | ------------------------------------- | ------------------------------------ |
| UI Streamlit   | http://localhost:8500                 | Chatbot d'exploration + prédiction   |
| API explore    | http://localhost:8081/explore         | Génération + exécution code pandas   |
| API predict    | http://localhost:8082/predict         | Prédiction du montant (modèle ML)    |
| MLflow         | http://localhost:5050                 | Tracking, registry modèles & prompts |
| Airflow        | http://localhost:8080                 | DAG `apd_pipeline` (`admin`/`admin`) |
| Dashboard Dash | http://localhost:8050                 | Visualisation des données APD        |

Le port hôte de l'API est **8081** par défaut. Surchargeable :
`API_PORT=9000 make up` (idem `UI_PORT`, `PREDICT_PORT`).

Arrêt : `make down`. Logs : `make logs`.

> **Pipeline Airflow** — l'orchestration horaire (téléchargement DagsHub,
> nettoyage, entraînement de 3 modèles, sélection du champion, alimentation de
> l'API explore) est documentée dans [data_airflow.md](data_airflow.md).
> Déclencher un run : `make airflow-trigger`.

### Sécurité d'exécution

Le code généré par le LLM est exécuté, donc encadré en profondeur :

1. **Validation AST** ([api/explore/sandbox.py](api/explore/sandbox.py)) avant toute exécution :
   imports limités à une whitelist (pandas, numpy, json…), blocage de
   `eval`/`exec`/`open`, des dunders et des méthodes d'écriture fichier
   (`to_csv`, `to_pickle`…). Testé : `make test` (18 cas).
2. **subprocess isolé** : `python -I`, avec un **timeout** (`EXEC_TIMEOUT`, 30 s).
3. **Volume données en lecture seule** (`./data:/data:ro`) : pas d'écriture ni
   d'exfiltration possible vers les données.

### Variables d'environnement (api)

| Variable               | Défaut                                          | Rôle                          |
| ---------------------- | ----------------------------------------------- | ----------------------------- |
| `DATA_PATH`            | `/data/processed/apd_explore.parquet`           | Fichier lu par le code généré |
| `LLM_URL`              | `http://host.docker.internal:8000/v1/chat/...`  | Endpoint du serveur LLM       |
| `LLM_MODEL`            | `mlx-community/Mistral-7B-Instruct-v0.3-4bit`   | Modèle servi                  |
| `MLFLOW_TRACKING_URI`  | `http://mlflow:5000`                            | Serveur MLflow (tracing/prompts) |
| `EXEC_TIMEOUT`         | `30`                                            | Timeout d'exécution (s)       |

## Arborescence

```
data/
  raw/        # CSV source APD (tracké par DVC)
  processed/  # apd_clean.parquet (modèle) + apd_explore.parquet (chatbot)
metrics/      # metrics/features.json (KPI du pipeline, versionné git)
models/
  data/       # modèle entraîné (pipeline.joblib, meta.json)
src/
  prepare.py  # stage prepare (dataset modèle)
  features.py # stage features
scripts/
  prepare_explore.py  # CSV brut -> apd_explore.parquet (dataset chatbot)
  train.py            # entraînement + logging MLflow
api/
  explore/    # FastAPI : génération + validation + exécution du code pandas
  predict/    # FastAPI : prédiction via le modèle ML
ui/           # Streamlit : interface de chat (app.py)
dags/         # Airflow : apd_pipeline (cf. data_airflow.md)
dashboard/    # Dash : visualisation des données (port 8050)
mlflow/       # serveur MLflow + prompts (register_prompts.py)
llm/          # serveur LLM hôte (start_vllm_mac.sh, .env.example)
docker-compose.yml  # api, ui, mlflow, postgres, airflow, dashboard
Makefile      # orchestration (make help)
dvc.yaml      # définition des stages
params.yaml   # hyperparamètres
dvc.lock      # hashes des inputs/outputs (versionné git, géré par DVC)
```
