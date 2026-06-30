# Pipeline de données — DAG Airflow `apd_pipeline`

Pipeline orchestré par Airflow qui, **toutes les heures**, récupère les données
APD depuis DagsHub, les transforme, entraîne trois modèles de régression
concurrents et promeut le meilleur. Le modèle sauvegardé alimente l'API
`predict` et le dashboard.

- **DAG** : [`dags/apd_dag.py`](dags/apd_dag.py)
- **Planification** : `@hourly` (`schedule_interval="@hourly"`, `catchup=False`)
- **Exécuteur** : `LocalExecutor` (backend PostgreSQL)
- **Tags** : `apd`, `ml`

---

## Vue d'ensemble

```
Branche ML :
download_data → clean_data → feature_engineering ─┬─→ train_linear  ─┐
                                                  ├─→ train_tree    ─┼─→ select_and_save
                                                  └─→ train_forest  ─┘

Branche données (alimente l'API explore) :
export_explore_parquet
```

Le DAG comporte **deux branches indépendantes** :

- **Branche ML** — les trois tâches d'entraînement s'exécutent **en parallèle**
  et transmettent leur score de validation croisée via **XCom**.
  `select_and_save` lit ces scores, choisit le meilleur modèle, le réentraîne
  sur l'ensemble des données et le sérialise.
- **Branche données** — `export_explore_parquet` produit le parquet consommé par
  l'**API explore** (chatbot). Elle ne dépend pas de la branche ML.

---

## Étapes du DAG

### 1. `download_data`

**Fonctionnel** — récupère le fichier source `aide-publique-au-developpement.csv`
(~106 000 lignes) depuis le stockage distant DagsHub.

**Technique**
- Client `boto3` configuré sur l'endpoint S3-compatible de DagsHub
  (`https://dagshub.com/llecorps/fev26_bmle_agent_apd.s3`).
- DVC stocke les fichiers **par hash MD5** (`files/md5/<2 premiers>/<reste>`),
  pas par nom logique. La clé est donc construite à partir du MD5 lu dans
  `data/raw/aide-publique-au-developpement.csv.dvc`.
- Credentials injectés via les variables d'environnement `DAGSHUB_ACCESS_KEY` /
  `DAGSHUB_SECRET_KEY`.
- Sortie : `/app/raw_files/apd_<YYYY-MM-DD_HH>.csv`.

### 2. `clean_data`

**Fonctionnel** — nettoie et filtre les données brutes pour ne garder que les
lignes exploitables.

**Technique**
- Lecture du dernier CSV de `raw_files/` (séparateur `;`).
- Strip des noms de colonnes ; suppression des lignes sans
  `Montant verse (K EUR)`, `Pays beneficiaire` ou `Secteur`.
- Filtre `Montant verse (K EUR) > 0` ; conversion de `Annee de declaration`
  en entier.
- Sortie : `/app/clean_data/apd_clean.csv`.

### 3. `feature_engineering`

**Fonctionnel** — construit la variable cible et sélectionne les features
pertinentes pour la modélisation.

**Technique**
- Cible : `log_montant = log1p("Montant verse (K EUR)")`.
- Features catégorielles retenues : `Agence`, `Type de financement`,
  `Pays beneficiaire`, `Région`, `Secteur`, `Catégorie CAD`, `Bi/Multi`.
- Feature numérique : `Annee de declaration`.
- Suppression des lignes incomplètes.
- Sortie : `/app/clean_data/apd_features.csv`.

### 4. `train_linear` / `train_tree` / `train_forest` (parallèle)

**Fonctionnel** — entraînent trois modèles de régression concurrents et
évaluent chacun par validation croisée.

| Tâche | Modèle | Hyperparamètres |
| --- | --- | --- |
| `train_linear` | `LinearRegression` | — |
| `train_tree`   | `DecisionTreeRegressor` | `max_depth=10` |
| `train_forest` | `RandomForestRegressor` | `n_estimators=100, max_depth=10` |

**Technique**
- Préprocessing commun via `ColumnTransformer` :
  - catégoriel → `SimpleImputer(constant)` + `OrdinalEncoder`
    (`handle_unknown="use_encoded_value"`),
  - numérique → `SimpleImputer(median)`.
- Évaluation : `cross_val_score` 5-fold, scoring `r2`.
- Le score R² moyen est poussé dans **XCom**
  (`xcom_push(key="score_<ModelName>")`).

#### Focus — transmission des scores via XCom

Chaque tâche d'entraînement tourne dans un **processus isolé** : elles ne
partagent ni variables Python ni mémoire. Pour que `select_and_save` puisse
comparer les trois modèles, il faut un canal de communication entre tâches —
c'est le rôle d'**XCom** (*cross-communication*).

**Qu'est-ce qu'XCom ?** Un mécanisme natif d'Airflow permettant à une tâche de
publier une petite valeur (chiffre, chaîne, dict…) que d'autres tâches du même
*DAG run* pourront relire. Techniquement, la valeur est **sérialisée et stockée
dans la base de métadonnées d'Airflow** (ici PostgreSQL), indexée par
`(dag_id, run_id, task_id, key)`. XCom est conçu pour de **petites** données
(scores, chemins, identifiants) — pas pour transporter un DataFrame ou un modèle.

**Côté producteur** — chaque tâche d'entraînement publie son score R² :

```python
def _train_model(model_name, **context):
    ...
    score = float(np.mean(cross_val_score(pipe, X, y, cv=5, scoring="r2")))
    context["ti"].xcom_push(key=f"score_{model_name}", value=score)
```

- `context["ti"]` est l'instance de tâche (*TaskInstance*) injectée par Airflow
  (grâce à `provide_context=True`).
- `xcom_push` écrit la paire `(key, value)` dans la base. Chaque modèle utilise
  une clé distincte : `score_LinearRegression`, `score_DecisionTreeRegressor`,
  `score_RandomForestRegressor`.

**Côté consommateur** — `select_and_save` relit les trois scores :

```python
ti = context["ti"]
scores = {
    "LinearRegression":      ti.xcom_pull(key="score_LinearRegression"),
    "DecisionTreeRegressor": ti.xcom_pull(key="score_DecisionTreeRegressor"),
    "RandomForestRegressor": ti.xcom_pull(key="score_RandomForestRegressor"),
}
best_name = max(scores, key=scores.get)   # meilleur R² CV
```

- `xcom_pull(key=...)` lit la valeur publiée plus tôt dans le **même run**.
- Grâce aux dépendances du DAG (`[t4a, t4b, t4c] >> t5`), Airflow garantit que
  les trois `xcom_push` ont eu lieu **avant** que `select_and_save` ne s'exécute.

**Pourquoi ce design ?** Il découple l'entraînement de la sélection : on peut
ajouter/retirer un modèle sans toucher à `select_and_save` (il suffit d'ajouter
une clé), et chaque entraînement reste parallélisable et ré-essayable
indépendamment.

### 5. `select_and_save`

**Fonctionnel** — sélectionne le modèle le plus performant, le réentraîne sur la
totalité des données et le sauvegarde pour la production.

**Technique**
- Lecture des trois scores via `xcom_pull` ; sélection par R² CV maximal.
- Split train/test (80/20) pour calculer les métriques finales
  (RMSE, MAE, R² sur le test set).
- Réentraînement du modèle gagnant sur **toutes** les données.
- Sorties :
  - `/app/models/data/pipeline.joblib` — pipeline sérialisé (joblib),
  - `/app/models/data/meta.json` — modèle choisi, scores CV, métriques test,
    liste des features, tailles train/test.

### `export_explore_parquet` (branche données)

**Fonctionnel** — alimente l'**API explore** (chatbot) avec des données à jour.
Le chatbot lit ce parquet à chaque requête pour exécuter le code pandas généré.

**Technique**
- Télécharge le **CSV propre** `aide-publique-au-developpement_clean.csv` depuis
  DagsHub (clé MD5 `e41e1c6c…`, distincte du CSV brut de la branche ML).
- Conversion CSV (séparateur `;`) → parquet **sans filtrage**.
- Sortie : `/app/data/processed/apd_clean.parquet` — monté en lecture seule côté
  `explore-api` (`./data:/data:ro`).
- Jeu de données : ~79 000 lignes, 50 colonnes (orienté engagements :
  `Engagements (K EUR)`, `log_engagements`, `Secteur`, `Pays beneficiaire`,
  `Région`, `Agence`…).

> **Note** — l'`explore-api` construit le schéma des colonnes injecté dans le
> prompt LLM **au démarrage**. Si le jeu de colonnes change, redémarrer le
> service : `docker compose restart explore-api`.

---

## Schéma d'exécution & dépendances

```python
# Branche ML
t1 >> t2 >> t3 >> [t4a, t4b, t4c] >> t5
# Branche données (indépendante)
t_explore
```

- Chaîne séquentielle jusqu'à `feature_engineering`.
- Fan-out vers les 3 entraînements parallèles.
- Fan-in vers `select_and_save` (s'exécute quand les 3 sont terminés).
- `default_args` : `retries=1`, `retry_delay=5 min`.

---

## Installation & démarrage d'Airflow

Airflow est intégré au `docker-compose.yml` (services `postgres`,
`airflow-init`, `airflow-webserver`, `airflow-scheduler`).

### Prérequis

- Docker + Docker Compose
- Variables DagsHub dans un fichier `.env` à la racine (gitignoré) :

```bash
DAGSHUB_ACCESS_KEY=<token DagsHub>
DAGSHUB_SECRET_KEY=<token DagsHub>
```

### Démarrage

```bash
# Charger les credentials DagsHub puis lancer toute la stack
set -a && source .env && set +a
make up            # ou : docker compose up -d
```

- **Airflow UI** : http://localhost:8080 — identifiants `admin` / `admin`
- Le DAG `apd_pipeline` est dé-pausé automatiquement
  (`AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'false'`).

### Dépendances Python des tâches

L'image `apache/airflow:2.9.1` ne contient pas les librairies ML. Elles sont
installées au démarrage des conteneurs via `_PIP_ADDITIONAL_REQUIREMENTS` :

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: boto3 pyarrow scikit-learn numpy==1.26.4
```

> **Note** : `numpy` est épinglé en `1.26.4` pour rester compatible (ABI) avec
> le `pandas` pré-installé dans l'image — une numpy 2.x casse l'import de pandas.

> Après modification de cette variable, recréer les conteneurs pour réinstaller :
> `docker compose up -d --force-recreate airflow-webserver airflow-scheduler`.

### Volumes montés (conteneurs Airflow)

| Hôte | Conteneur | Rôle |
| --- | --- | --- |
| `./dags` | `/opt/airflow/dags` | Définitions des DAG |
| `./logs` | `/opt/airflow/logs` | Logs des tâches |
| `./plugins` | `/opt/airflow/plugins` | Plugins Airflow |
| `./raw_files` | `/app/raw_files` | CSV bruts téléchargés |
| `./clean_data` | `/app/clean_data` | Données nettoyées / features |
| `./models/data` | `/app/models/data` | Modèle + métadonnées produits |
| `./data` | `/app/data` | Données du repo (parquet, etc.) |

### Déclencher / surveiller le DAG

```bash
# Déclencher un run manuel
docker compose exec airflow-scheduler airflow dags trigger apd_pipeline

# Lister les runs
docker compose exec airflow-scheduler airflow dags list-runs -d apd_pipeline

# Logs d'une tâche (ex. download_data)
docker compose exec airflow-scheduler \
  airflow tasks logs apd_pipeline download_data <run_id>
```

Depuis l'UI : ouvrir le DAG `apd_pipeline`, onglet **Graph**, puis **Clear** ou
**Trigger** pour (re)lancer un run.
