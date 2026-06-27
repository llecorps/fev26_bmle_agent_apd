from fastapi import FastAPI, Header
from pydantic import BaseModel
import requests, re
import tempfile
import subprocess
import sys
import os
import mlflow
import json

from sandbox import validate_code, CodeValidationError


api = FastAPI(
    title="REST API",
    description="API powered by FastAPI.",
    version="0.0.1")

# ─── CONFIGURATION MLFLOW ───────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("explore-llm-chat")
# ────────────────────────────────────────────────────────────

# Fichier de données exposé au code généré. Par défaut la sortie du pipeline DVC
# (data/processed/apd_clean.parquet) montée dans le conteneur sous /data.
DATA_PATH = os.getenv("DATA_PATH", "/data/processed/apd_clean.parquet")

# Endpoint du serveur LLM (vLLM/MLX)
LLM_URL = os.getenv("LLM_URL", "http://host.docker.internal:8000/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Délai max d'exécution du code généré avant kill du subprocess (secondes).
EXEC_TIMEOUT = float(os.getenv("EXEC_TIMEOUT", "30"))

# Nombre total de tentatives LLM (1 génération + (N-1) réparations sur erreur).
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))


def _build_schema(data_path: str, enum_max: int = 25) -> str:
    """Décrit les colonnes du dataset pour les injecter dans le prompt.

    Sans ce schéma, un LLM 7B invente des noms de colonnes (ex. 'agency' au
    lieu de 'Agence'). On liste toutes les colonnes, et pour les catégorielles
    à faible cardinalité on énumère les valeurs exactes.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
    except Exception as exc:  # fichier absent (tests) -> pas de schéma
        print(f"Schéma indisponible ({exc})")
        return ""

    lines = []
    for col in df.columns:
        s = df[col]
        if s.dtype == object or str(s.dtype) == "string":
            nunique = s.nunique(dropna=True)
            if nunique <= enum_max:
                vals = ", ".join(repr(v) for v in sorted(s.dropna().unique()))
                lines.append(f"- {col!r} (texte) valeurs: {vals}")
            else:
                lines.append(f"- {col!r} (texte, {nunique} valeurs distinctes)")
        else:
            lines.append(f"- {col!r} ({s.dtype})")
    return "\n    ".join(lines)


# Schéma construit une fois au démarrage (le parquet est monté dans le conteneur).
SCHEMA_TEXT = _build_schema(DATA_PATH)


class ChatRequest(BaseModel):
    message: str


def run_sandboxed(code: str):
    """Valide (AST) puis exécute le code dans un subprocess isolé avec timeout.

    Retourne (returncode, stdout, stderr). returncode == -1 pour un rejet
    sandbox ou un timeout (pas d'exécution réelle).
    """
    try:
        validate_code(code)
    except CodeValidationError as exc:
        print(f"Code rejeté par la sandbox : {exc}")
        return -1, "", f"Code rejeté par la sandbox : {exc}"

    with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+t') as temp_file:
        temp_file.write(code)
        temp_file.flush()
        print(f"Exécution du fichier temporaire : {temp_file.name}")
        # `-I` : mode isolé. `timeout` : tue le process s'il boucle ou traîne.
        try:
            r = subprocess.run([sys.executable, "-I", temp_file.name],
                               capture_output=True, text=True, timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"Exécution interrompue après {EXEC_TIMEOUT:.0f}s (timeout)")
            return -1, "", f"Exécution interrompue après {EXEC_TIMEOUT:.0f}s (timeout)"

    return r.returncode, r.stdout, r.stderr

@mlflow.trace(name="explore-llm-session", span_type="CHAIN")
@api.post('/explore')
async def post_explore(request: ChatRequest):

    """Returns explored data.
    """

    print(f"Requête reçue : {request.message}")

    # Enregistrer l'input de l'utilisateur dans la trace actuelle
    span = mlflow.get_current_active_span()
    if span:
        span.set_inputs({"user_query": request.message})

    code = generate_code_with_llm(request.message, span)
    returncode, stdout, stderr = -1, "", ""

    # Boucle d'auto-réparation : si l'exécution échoue, on renvoie l'erreur au
    # LLM pour qu'il corrige (un modèle 7B oublie parfois un import ou un nom).
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Tentative {attempt}/{MAX_ATTEMPTS} ---")
        returncode, stdout, stderr = run_sandboxed(code)
        if returncode == 0:
            print(stdout)
            break
        print(f"Échec (returncode={returncode}) : {stderr.strip()[-300:]}")
        if attempt < MAX_ATTEMPTS:
            code = repair_code_with_llm(request.message, code, stderr, span)

    # Enregistrer le résultat ou l'erreur à la racine de la trace
    if span:
        span.set_outputs({"result": stdout, "error": stderr, "success": returncode == 0})

    return {'result': stdout, 'error': stderr, 'returncode': returncode}

@mlflow.trace(name="vllm-generation", span_type="LLM")
def _call_llm(prompt: str) -> str:
    """Appelle le serveur LLM (API OpenAI-compatible) et renvoie le code extrait."""
    r = requests.post(LLM_URL,
                     headers={'Authorization': 'Bearer token', 'Content-Type': 'application/json'},
                     json={
        "model": LLM_MODEL,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=120)
    response_dict = r.json()

    # ─── EXTRACTION DES MÉTRIQUES DE TOKENS LLM ───
    try:
        span = mlflow.get_current_active_span()
        if span:
            # On log le prompt exact envoyé et le modèle utilisé
            span.set_inputs({"prompt": prompt, "model": LLM_MODEL})
            
            # vLLM renvoie une structure standardisée "usage"
            usage = response_dict.get("usage", {})
            span.set_attributes({
                "llm.model": LLM_MODEL,
                "llm.usage.prompt_tokens": usage.get("prompt_tokens", 0),
                "llm.usage.completion_tokens": usage.get("completion_tokens", 0),
                "llm.usage.total_tokens": usage.get("total_tokens", 0)
            })
    except Exception as e:
        print(f"⚠️ Impossible de rattacher les métriques à la trace MLflow : {e}")
    # ───────────────────────────────────────────────

    generated_code = extract_pure_code(response_dict['choices'][0]['message']['content'])
    print(generated_code)
    return generated_code


def generate_code_with_llm(chat_request: str, root_span=None) -> str:
    prompt_name = "explore-code-generation-prompt"
    
    try:
        # Charger la version 'champion' depuis le Prompt Registry de MLflow
        mlflow_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@champion")
        template_str = mlflow_prompt.template

        # Assigner le prompt à la trace active
        # Cela garantit que la trace apparaîtra dans l'onglet "Traces"
        mlflow.update_current_trace(
            tags={
                "mlflow.linkedPrompts": json.dumps([
                    {
                        "name": mlflow_prompt.name, 
                        "version": str(mlflow_prompt.version)
                    }
                ])
            }
        )
   
    except Exception as e:
        print(f"⚠️ Impossible de charger le prompt depuis MLflow ({e}).")
        raise e

    # Remplacement des variables du template MLflow
    prompt = (
        template_str.replace("{{data_path}}", DATA_PATH)
        .replace("{{schema}}", SCHEMA_TEXT)
        .replace("{{question}}", chat_request)
    )

    print("Prompt envoyé au modèle :", prompt[:500], "...")
    return _call_llm(prompt)


def repair_code_with_llm(chat_request: str, broken_code: str, error: str, root_span=None) -> str:
    """Renvoie au LLM le code fautif et son erreur d'exécution pour correction."""
    prompt_name = "explore-code-repair-prompt"

    try:
        # Charger la version 'champion' depuis le Prompt Registry de MLflow
        mlflow_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@champion")
        template_str = mlflow_prompt.template

        # Assigner le prompt à la trace active
        # Cela garantit que la trace apparaîtra dans l'onglet "Traces"
        mlflow.update_current_trace(
            tags={
                "mlflow.linkedPrompts": json.dumps([
                    {
                        "name": mlflow_prompt.name, 
                        "version": str(mlflow_prompt.version)
                    }
                ])
            }
        )

    except Exception as e:
        print(f"⚠️ Impossible de charger le prompt depuis MLflow ({e}).")
        raise e

    # Remplacement des variables du template MLflow
    prompt = (
        template_str.replace("{{data_path}}", DATA_PATH)
        .replace("{{schema}}", SCHEMA_TEXT)
        .replace("{{question}}", chat_request)
        .replace("{{broken_code}}", broken_code)
        .replace("{{error}}", error)
    )

    print("Réparation demandée au modèle.")
    return _call_llm(prompt)


def extract_pure_code(llm_response: str) -> str:
    # Recherche le bloc de code ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Si le modèle a oublié les backticks mais a craché du code
    return llm_response.strip()
