# Makefile d'orchestration de l'environnement APD.
# Trois briques : pipeline DVC (données) · serveur LLM · services Docker (api + ui).

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
DVC     := .venv/bin/dvc
DATA    := data/processed/apd_clean.parquet
COMPOSE := docker compose

# URI du serveur MLflow (port hôte mappé dans docker-compose). Surchargeable :
#   make init-prompts MLFLOW_URI=http://autre-hote:5050
MLFLOW_URI ?= http://localhost:5050

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Aide
# ---------------------------------------------------------------------------
.PHONY: help
help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environnement Python + données (pipeline DVC)
# ---------------------------------------------------------------------------
.PHONY: install
install:  ## Crée le venv et installe les dépendances du pipeline
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: data
data: $(DATA)  ## Construit les données (dvc repro) si nécessaire

$(DATA):
	$(DVC) repro

.PHONY: repro
repro:  ## Force le rejeu complet du pipeline DVC
	$(DVC) repro --force

.PHONY: pull
pull:  ## Récupère les données depuis le remote DagsHub
	$(DVC) pull

.PHONY: push
push:  ## Envoie les données vers le remote DagsHub
	$(DVC) push

.PHONY: metrics
metrics:  ## Affiche les métriques du pipeline
	$(DVC) metrics show

# ---------------------------------------------------------------------------
# Serveur LLM (vllm-mlx sur Apple Silicon) — process hôte, à lancer à part
# ---------------------------------------------------------------------------
.PHONY: llm
llm:  ## Démarre le serveur LLM local (port 8000, lit llm/.env)
	bash llm/start_vllm_mac.sh

# ---------------------------------------------------------------------------
# Services Docker (api + ui + mlflow + airflow + dashboard)
# ---------------------------------------------------------------------------
.PHONY: up
up: $(DATA) monitoring-data  ## Build + démarre TOUTE la stack (api, ui, mlflow, airflow, dashboard, monitoring)
	$(COMPOSE) up --build -d
	@echo "UI         : http://localhost:$${UI_PORT:-8500}"
	@echo "API        : http://localhost:$${API_PORT:-8081}/explore"
	@echo "Predict API: http://localhost:$${PREDICT_PORT:-8082}/metrics"
	@echo "MLflow     : http://localhost:5050"
	@echo "Airflow    : http://localhost:8080  (admin/admin)"
	@echo "Dashboard  : http://localhost:8050"
	@echo "Prometheus : http://localhost:9090"
	@echo "Grafana    : http://localhost:3000  (admin/admin)"
	@echo "Rappel: le serveur LLM doit tourner ('make llm' dans un autre terminal)."

.PHONY: down
down:  ## Arrête et supprime les conteneurs
	$(COMPOSE) down

.PHONY: logs
logs:  ## Suit les logs des conteneurs
	$(COMPOSE) logs -f

# ---------------------------------------------------------------------------
# Monitoring (Prometheus + Grafana + Evidently)
# ---------------------------------------------------------------------------
MONITORING_DATA := data/monitoring/reference_january.parquet

.PHONY: monitoring-data
monitoring-data: $(MONITORING_DATA)  ## Prépare les données de référence/courantes (/evaluate)

$(MONITORING_DATA):
	$(PYTHON) scripts/prepare_monitoring.py

.PHONY: evaluate
evaluate:  ## Déclenche /evaluate sur la predict-api (met à jour les métriques ML)
	curl -s -X POST http://localhost:$${PREDICT_PORT:-8082}/evaluate \
		-H "Content-Type: application/json" -d '{}' | python3 -m json.tool

# ---------------------------------------------------------------------------
# Airflow (DAG apd_pipeline)
# ---------------------------------------------------------------------------
.PHONY: airflow-trigger
airflow-trigger:  ## Déclenche manuellement le DAG apd_pipeline
	$(COMPOSE) exec airflow-scheduler airflow dags trigger apd_pipeline

.PHONY: airflow-runs
airflow-runs:  ## Liste les runs du DAG apd_pipeline
	$(COMPOSE) exec airflow-scheduler airflow dags list-runs -d apd_pipeline

.PHONY: airflow-logs
airflow-logs:  ## Suit les logs des conteneurs Airflow
	$(COMPOSE) logs -f airflow-scheduler airflow-webserver

# ---------------------------------------------------------------------------
# Prompts (MLflow Prompt Registry)
# ---------------------------------------------------------------------------
.PHONY: init-prompts
init-prompts:  ## Enregistre les prompts (génération de code + réparation de code) et bascule l'alias @champion
	@echo "Enregistrement des prompts dans MLflow ($(MLFLOW_URI))…"
	@echo "Rappel : la stack doit tourner ('make up') pour que MLflow réponde."
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) mlflow/register_prompts.py

# ---------------------------------------------------------------------------
# Qualité
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Lance les tests (sandbox de l'API)
	cd api && ../.venv/bin/python -m pytest test_sandbox.py -v

.PHONY: clean
clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache api/.pytest_cache