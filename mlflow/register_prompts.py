"""Enregistre les prompts (génération + réparation) dans le Prompt Registry
MLflow et bascule l'alias `@champion` sur la nouvelle version.

Usage :
    # depuis l'hôte (port mappé 5050 -> 5000 dans docker-compose)
    MLFLOW_TRACKING_URI=http://localhost:5050 python mlflow/register_prompts.py

    # ou depuis le réseau docker
    MLFLOW_TRACKING_URI=http://mlflow:5000 python mlflow/register_prompts.py
"""

import os
import pathlib
import mlflow

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow.set_tracking_uri(MLFLOW_URI)

# Rattacher les prompts à la même expérience que le chatbot (api/explore/explore.py
# fait mlflow.set_experiment("explore-llm-chat")). 
# sans cette ligne, il atterrit dans l'expérience "Default".
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "explore-llm-chat")
mlflow.set_experiment(EXPERIMENT)

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "mlflow" / "prompts"

# (nom enregistré dans MLflow, fichier source) — les NOMS doivent correspondre
# EXACTEMENT à ceux chargés par api/explore/explore.py.
PROMPTS = [
    ("explore-code-generation-prompt", "explore-code-generation-prompt.txt",
     "sortie JSON structurée (type/labels/values) pour rendu graphique + textuel."),
    ("explore-code-repair-prompt", "explore-code-repair-prompt.txt",
     "placeholders alignés sur explore.py, rendu graphique."),
]


def main() -> None:
    print(f"MLflow tracking URI : {MLFLOW_URI}")
    for name, filename, message in PROMPTS:
        template = (PROMPTS_DIR / filename).read_text(encoding="utf-8")
        pv = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=message,
        )
        mlflow.genai.set_prompt_alias(name=name, alias="champion", version=pv.version)
        print(f"✓ {name} -> version {pv.version} (alias @champion)")


if __name__ == "__main__":
    main()