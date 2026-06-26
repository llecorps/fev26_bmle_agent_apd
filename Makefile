# Makefile d'orchestration de l'environnement APD.
# Trois briques : pipeline DVC (données) · serveur LLM · services Docker (api + ui).

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
DVC     := .venv/bin/dvc
DATA    := data/processed/apd_clean.parquet
COMPOSE := docker compose

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
# Serveur LLM (Ollama — géré par Docker Compose)
# ---------------------------------------------------------------------------
.PHONY: llm-pull
llm-pull:  ## Télécharge le modèle mistral:7b dans le conteneur ollama
	$(COMPOSE) exec ollama ollama pull mistral:7b

# ---------------------------------------------------------------------------
# Services Docker (chatbot + airflow + dashboard, un seul compose)
# ---------------------------------------------------------------------------
.PHONY: up
up: $(DATA)  ## Build + démarre tous les services (chatbot + airflow + dashboard)
	$(COMPOSE) up --build -d
	@echo "Attente du démarrage d'Ollama..."
	@sleep 5
	@$(MAKE) llm-pull
	@echo "UI        : http://localhost:$${UI_PORT:-8500}"
	@echo "API       : http://localhost:$${API_PORT:-8081}/explore"
	@echo "Airflow   : http://localhost:8080  (admin/admin)"
	@echo "Dashboard : http://localhost:8050"

.PHONY: down
down:  ## Arrête et supprime les conteneurs
	$(COMPOSE) down

.PHONY: logs
logs:  ## Suit les logs des conteneurs
	$(COMPOSE) logs -f

# ---------------------------------------------------------------------------
# Qualité
# ---------------------------------------------------------------------------
.PHONY: test
test:  ## Lance les tests (sandbox de l'API)
	cd api/explore && ../../.venv/bin/python -m pytest test_sandbox.py -v

.PHONY: clean
clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache api/.pytest_cache
