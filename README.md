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
Streamlit (ui)  ──HTTP──►  FastAPI /explore  ──HTTP──►  Ollama (mistral:7b)
   :8500                      :8080 (interne)             :11434 (interne)
                                │
                                ▼
                  1. génère le code pandas (LLM)
                  2. validation AST (api/explore/sandbox.py)
                  3. exécution sandboxée (python -I, timeout)
                  4. lit data/processed/apd_clean.parquet (monté :ro)
```

Les trois services (`ollama`, `explore-api`, `ui`) sont gérés par docker compose.
Le modèle `mistral:7b` est téléchargé automatiquement au premier `make up`.

### Démarrer le chatbot

```bash
# 1. construire les données si besoin
make data            # ou : make pull

# 2. démarrer tous les services + télécharger le modèle
make up
```

- UI : http://localhost:8500
- API explore : http://localhost:8081/explore
- API predict : http://localhost:8082/predict

Le port hôte de l'API est **8081** par défaut. Surchargeable : `API_PORT=9000 make up`
(idem `UI_PORT`, `PREDICT_PORT`).

Arrêt : `make down`. Logs : `make logs`.

### Sécurité d'exécution

Le code généré par le LLM est exécuté, donc encadré en profondeur :

1. **Validation AST** ([api/explore/sandbox.py](api/explore/sandbox.py)) avant toute exécution :
   imports limités à une whitelist (pandas, numpy, json…), blocage de
   `eval`/`exec`/`open`, des dunders et des méthodes d'écriture fichier
   (`to_csv`, `to_pickle`…). Testé : `make test` (18 cas).
2. **subprocess isolé** : `python -I`, avec un **timeout** (`EXEC_TIMEOUT`, 30 s).
3. **Volume données en lecture seule** (`./data:/data:ro`) : pas d'écriture ni
   d'exfiltration possible vers les données.

### Variables d'environnement (explore-api)

| Variable       | Défaut                                              | Rôle                          |
| -------------- | --------------------------------------------------- | ----------------------------- |
| `DATA_PATH`    | `/data/processed/apd_clean.parquet`                 | Fichier lu par le code généré |
| `LLM_URL`      | `http://ollama:11434/v1/chat/completions`           | Endpoint Ollama               |
| `LLM_MODEL`    | `mistral:7b`                                        | Modèle servi                  |
| `EXEC_TIMEOUT` | `30`                                                | Timeout d'exécution (s)       |

## Sécurité — API Key

Les endpoints `/explore` et `/predict` supportent une authentification par clé API via le header `X-API-Key`.

**En développement local** — auth désactivée par défaut, aucune configuration requise :

```bash
make up
```

**Avec auth activée** (recommandé pour un déploiement ou une démo publique) :

```bash
API_KEY=mon-secret make up
```

Toute requête sans le header correct reçoit un `403 Forbidden` :

```bash
# ✅ Requête valide
curl -X POST http://localhost:8081/explore \
     -H "X-API-Key: mon-secret" \
     -H "Content-Type: application/json" \
     -d '{"message": "Quels sont les 5 pays qui reçoivent le plus d'aide ?"}'

# ❌ Sans clé → 403
curl -X POST http://localhost:8081/explore \
     -H "Content-Type: application/json" \
     -d '{"message": "..."}'
```

L'UI Streamlit propage automatiquement la clé — aucune configuration supplémentaire nécessaire côté interface.

> Ne jamais commiter la valeur de `API_KEY`. La passer uniquement via variable d'environnement ou un fichier `.env` gitignoré.

## Commandes make

```
make install     Crée le venv et installe les dépendances du pipeline
make data        Construit les données (dvc repro) si nécessaire
make repro       Force le rejeu complet du pipeline DVC
make pull        Récupère les données depuis le remote DagsHub
make push        Envoie les données vers le remote DagsHub
make metrics     Affiche les métriques du pipeline
make llm-pull    Télécharge le modèle mistral:7b dans le conteneur ollama
make up          Build + démarre tous les services (ollama + api + ui)
make down        Arrête et supprime les conteneurs
make logs        Suit les logs des conteneurs
make test        Lance les tests (sandbox de l'API)
make clean       Supprime les caches Python
```

## Arborescence

```
data/
  raw/        # CSV source APD (tracké par DVC)
  processed/  # apd_clean.parquet (sortie de prepare)
metrics/      # metrics/features.json (KPI du pipeline, versionné git)
models/
  data/       # pipeline.joblib + meta.json + dropdowns.json
src/
  prepare.py  # stage prepare
  features.py # stage features
api/
  explore/    # FastAPI : génération + validation + exécution du code pandas
  predict/    # FastAPI : prédiction via le modèle ML
ui/           # Streamlit : interface de chat (app.py)
llm/          # .env.example (config Ollama optionnelle)
docker-compose.yml  # services ollama + explore-api + predict-api + ui
Makefile      # orchestration (make help)
dvc.yaml      # définition des stages
params.yaml   # hyperparamètres
dvc.lock      # hashes des inputs/outputs (versionné git, géré par DVC)
```
