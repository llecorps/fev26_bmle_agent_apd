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
# Serveur LLM (vllm-mlx sur Apple Silicon) — process hôte, à lancer à part
# ---------------------------------------------------------------------------
.PHONY: llm
llm:  ## Démarre le serveur LLM local (port 8000, lit llm/.env)
	bash llm/start_vllm_mac.sh

# ---------------------------------------------------------------------------
# Services Docker (api + ui)
# ---------------------------------------------------------------------------
.PHONY: up
up: $(DATA)  ## Build + démarre l'API et l'UI (nécessite les données et le LLM)
	$(COMPOSE) up --build -d
	@echo "UI    : http://localhost:8500"
	@echo "API   : http://localhost:8080/explore"
	@echo "Rappel: le serveur LLM doit tourner ('make llm' dans un autre terminal)."

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
	cd api && ../.venv/bin/python -m pytest test_sandbox.py -v

.PHONY: clean
clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache api/.pytest_cache
