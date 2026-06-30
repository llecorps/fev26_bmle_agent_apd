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

## Dashboard (Dash / Plotly)

Application **Dash** servie sur http://localhost:8050, alimentée par le DAG
Airflow (cf. [data_airflow.md](data_airflow.md)).

```
DAG Airflow ──► clean_data/apd_clean.csv ──┐
DAG Airflow ──► models/data/meta.json ─────┴──► Dashboard Dash (:8050)
                                                KPIs · graphiques · carte modèle
```

**Fonctionnement** ([dashboard/app.py](dashboard/app.py)) :

- **Sources** (montées en lecture seule) : `clean_data/apd_clean.csv` (données
  nettoyées par le DAG) et `models/data/meta.json` (métriques du modèle champion).
- **Layout** : barre de filtres (année, région, secteur) + KPIs (montant total,
  nb projets, pays, ticket moyen) + 4 graphiques (top 10 pays, secteurs,
  évolution annuelle, montants par région) + carte du modèle actif.
- **Réactivité** : les *callbacks* Dash (`@app.callback(Output, Input)`) se
  ré-exécutent dès qu'un filtre change — ils relisent le CSV, l'agrègent et
  renvoient les figures Plotly.
- **Auto-refresh** : un `dcc.Interval` recharge les données **toutes les 60 s**,
  donc le dashboard reflète automatiquement le dernier run du DAG.

| Variable    | Défaut             | Rôle                          |
| ----------- | ------------------ | ----------------------------- |
| `CLEAN_DIR` | `/app/clean_data`  | Dossier du CSV nettoyé        |
| `MODEL_DIR` | `/app/models/data` | Dossier des métriques modèle  |

## Commandes make

```
make install         Crée le venv et installe les dépendances du pipeline
make data            Construit les données (dvc repro) si nécessaire
make repro           Force le rejeu complet du pipeline DVC
make pull            Récupère les données depuis le remote DagsHub
make push            Envoie les données vers le remote DagsHub
make metrics         Affiche les métriques du pipeline
make llm             Démarre le serveur LLM local (port 8000, lit llm/.env)
make init-prompts    Enregistre les prompts (génération + réparation) et bascule l'alias @champion
make up              Build + démarre TOUTE la stack (api, ui, mlflow, airflow, dashboard)
make down            Arrête et supprime les conteneurs
make logs            Suit les logs des conteneurs
make airflow-trigger Déclenche manuellement le DAG apd_pipeline
make airflow-runs    Liste les runs du DAG apd_pipeline
make airflow-logs    Suit les logs des conteneurs Airflow
make test            Lance les tests (sandbox de l'API)
make clean           Supprime les caches Python
make help            Affiche l'aide
```

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
