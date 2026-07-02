# Monitoring — Prometheus · Grafana · Evidently

Surveillance des **performances du modèle**, des **dérives de données** et de
l'**infrastructure**, avec dashboards Grafana provisionnés automatiquement.

## Stack

| Service         | Port | Rôle                                             |
| --------------- | ---- | ------------------------------------------------ |
| Prometheus      | 9090 | Collecte des métriques (scrape) + règles d'alerte |
| Grafana         | 3000 | Dashboards (as code) — `admin`/`admin`           |
| node-exporter   | 9100 | Métriques infra hôte (CPU, RAM, disque)          |
| predict-api     | 8082 | Expose `/metrics` (app + modèle) et `/evaluate`  |

## Démarrage (reproductible)

```bash
make up          # prépare les données monitoring, build + démarre toute la stack
make evaluate    # calcule RMSE/MAE/R² + dérive -> alimente les gauges Prometheus
```

`make up` exécute d'abord `monitoring-data` (génère `data/monitoring/*.parquet`
depuis le CSV brut) puis lance les conteneurs. Grafana charge automatiquement la
datasource Prometheus et les 3 dashboards au démarrage — aucune action manuelle.

## Métriques exposées par la predict-api (`/metrics`)

| Métrique                             | Type      | Labels                          | Source        |
| ------------------------------------ | --------- | ------------------------------- | ------------- |
| `api_requests_total`                 | Counter   | endpoint, method, status_code   | middleware    |
| `api_request_duration_seconds`       | Histogram | endpoint, method, status_code   | middleware    |
| `model_rmse_score`                   | Gauge     | —                               | `/evaluate`   |
| `model_mae_score`                    | Gauge     | —                               | `/evaluate`   |
| `model_r2_score`                     | Gauge     | —                               | `/evaluate`   |
| `model_dataset_drift_share`          | Gauge     | —                               | `/evaluate`   |
| `evidently_data_drift_detected_status` | Gauge   | —                               | `/evaluate`   |

### Métrique personnalisée : `model_dataset_drift_share`

Part des features en dérive (0..1), calculée par Evidently (`DataDriftPreset`)
entre la référence (**janvier**) et les données courantes (**février**).

**Pourquoi c'est pertinent** : pour ce modèle de régression, les vraies
étiquettes (montants réellement engagés) n'arrivent qu'avec un **retard** de
plusieurs semaines/mois. On ne peut donc pas recalculer le RMSE en continu sur
des données fraîches. La **dérive des features** est l'**indicateur avancé** :
elle prévient d'une dégradation *probable* des performances **avant** qu'on
puisse la mesurer sur des labels réels. Suivre la *part* de features en dérive
(plutôt qu'un simple booléen) donne en plus une notion de **gravité**.

## Endpoint `/evaluate`

Compare des données courantes à la référence de janvier :

1. charge la référence `data/monitoring/reference_january.parquet` ;
2. charge les données courantes (corps de requête `current_data`, sinon
   `current_february.parquet` par défaut) ;
3. prédit avec le modèle et calcule **RMSE/MAE/R²** (sklearn, échelle log1p) ;
4. exécute **Evidently** (`DataDriftPreset`) pour la dérive des features ;
5. met à jour les Gauges Prometheus et renvoie un `EvaluationReportOutput`.

```bash
# période par défaut (février)
curl -X POST localhost:8082/evaluate -H 'Content-Type: application/json' -d '{}'

# données courantes fournies explicitement
curl -X POST localhost:8082/evaluate -H 'Content-Type: application/json' \
  -d '{"current_data": [{"Agence": "...", "Engagements (K EUR)": 1200, ...}]}'
```

## Dashboards Grafana (as code)

Provisionnés depuis `deployment/grafana/` — dossier `APD` dans Grafana :

- **API Performance** — taux de requêtes, latence P95, taux d'erreur 5xx, requêtes par statut.
- **Model Performance & Drift** — RMSE / MAE / R², part de features en dérive, statut de dérive.
- **Infrastructure Overview** — CPU, RAM, disque (node-exporter).

## Alerting

- **Prometheus** (`deployment/prometheus/rules/alert_rules.yml`) :
  - `PredictApiDown` — l'API ne répond plus au scrape depuis 1 min.
  - `PredictApiHighErrorRate` — > 5 % d'erreurs 5xx sur 5 min.
  - `ModelDataDriftDetected` — dérive globale du dataset détectée.
- **Grafana** — une alerte sur métrique ML à créer dans l'UI (ex. `model_rmse_score`
  > seuil, ou `model_dataset_drift_share` > 0.5). Voir la section « Alerting Grafana »
  ci-dessous.

### Créer l'alerte ML dans Grafana (UI)

1. *Alerting → Alert rules → New alert rule*.
2. Datasource **Prometheus**, requête `model_rmse_score`.
3. Condition : `IS ABOVE 0.5` (seuil à ajuster selon le RMSE de référence).
4. Évaluation toutes les 1 min, `for: 5m`, sévérité *warning*.
5. Enregistrer — l'alerte se déclenche si le RMSE dépasse durablement le seuil.

## Structure

```
deployment/
  prometheus/
    prometheus.yml              # scrape predict-api + node-exporter
    rules/alert_rules.yml       # règles d'alerte
  grafana/
    provisioning/
      datasources/datasources.yaml   # datasource Prometheus
      dashboards/dashboards.yaml      # provider "dashboards as code"
    dashboards/
      api_performance.json
      model_performance_drift.json
      infrastructure_overview.json
```
