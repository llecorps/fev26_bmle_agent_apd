from fastapi import FastAPI, Header
from pydantic import BaseModel
import requests, re
import tempfile
import subprocess
import sys
import os
import mlflow
import json

from sandbox import validate_code, sanitize_code, CodeValidationError


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

# Plafond de tokens générés par le LLM. Le code pandas attendu est court : 512
# suffit largement et c'est ~2x plus rapide qu'à 1024 sur un 7B local. (Sans
# limite explicite, certains serveurs tronquent -> "unterminated string".)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Délai max d'un appel au serveur LLM (secondes). À garder cohérent avec le
# timeout de l'UI : UI_timeout >= MAX_ATTEMPTS * LLM_TIMEOUT + MAX_ATTEMPTS * EXEC_TIMEOUT.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))


# Réglages du schéma injecté dans le prompt. Un schéma trop long ralentit
# fortement la génération sur un 7B local (prefill) -> on l'allège.
SCHEMA_ENUM_MAX = int(os.getenv("SCHEMA_ENUM_MAX", "12"))   # énumère si <= N valeurs
SCHEMA_MAX_VALS = int(os.getenv("SCHEMA_MAX_VALS", "12"))   # nb max de valeurs listées
SCHEMA_VAL_MAXLEN = int(os.getenv("SCHEMA_VAL_MAXLEN", "40"))  # tronque chaque valeur


def _short(v, maxlen: int = SCHEMA_VAL_MAXLEN) -> str:
    """Représentation courte d'une valeur (guillemets + troncature)."""
    s = str(v)
    if len(s) > maxlen:
        s = s[:maxlen - 1] + "…"
    return repr(s)


def _build_schema(data_path: str, enum_max: int = SCHEMA_ENUM_MAX) -> str:
    """Décrit les colonnes du dataset pour les injecter dans le prompt.

    Sans ce schéma, un LLM 7B invente des noms de colonnes (ex. 'agency' au
    lieu de 'Agence'). On liste toutes les colonnes ; pour les catégorielles à
    faible cardinalité ET aux libellés courts on énumère quelques valeurs, sinon
    on donne juste le nombre de modalités. Objectif : schéma compact pour ne pas
    ralentir la génération.
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
            uniques = s.dropna().unique()
            nunique = len(uniques)
            avg_len = (sum(len(str(v)) for v in uniques) / nunique) if nunique else 0
            # On énumère seulement si peu de modalités ET libellés courts.
            if nunique <= enum_max and avg_len <= SCHEMA_VAL_MAXLEN:
                vals = ", ".join(_short(v) for v in sorted(uniques))
                lines.append(f"- {col!r} (texte) valeurs: {vals}")
            else:
                ex = ", ".join(_short(v) for v in list(uniques)[:3])
                lines.append(f"- {col!r} (texte, {nunique} modalités, ex: {ex})")
        else:
            lines.append(f"- {col!r} ({s.dtype})")
    return "\n    ".join(lines)


# Schéma construit une fois au démarrage (le parquet est monté dans le conteneur).
SCHEMA_TEXT = _build_schema(DATA_PATH)


class ChatRequest(BaseModel):
    message: str


# ─── RÉSUMÉ TEXTUEL DU RÉSULTAT ─────────────────────────────
# Le code généré renvoie une donnée structurée (cf. contrat dans le prompt). On
# en dérive ici une phrase de synthèse FR, déterministe (jamais hallucinée, zéro
# appel LLM supplémentaire). L'UI affiche ce texte + le graphique.

def _fmt(n) -> str:
    """Formate un nombre à la française : 1 234 567,89 -> '1 234 567,89'."""
    try:
        f = float(n)
    except (TypeError, ValueError):
        return str(n)
    if f == int(f):
        s = f"{int(f):,}".replace(",", " ")
    else:
        s = f"{f:,.2f}".replace(",", " ").replace(".", ",")
    return s


def summarize_payload(payload) -> str:
    """Construit une synthèse en français à partir du payload structuré."""
    if not isinstance(payload, dict):
        return str(payload)[:500]

    ptype = payload.get("type")
    title = payload.get("title", "")
    unit = payload.get("unit", "") or ""
    unit_s = f" {unit}".rstrip()

    if ptype in ("bar", "line", "pie"):
        labels = payload.get("labels", []) or []
        values = payload.get("values", []) or []
        if not labels or not values:
            return title or "Aucune donnée à afficher."
        parts = [f"**{title}**." if title else ""]
        top_lbl, top_val = labels[0], values[0]
        parts.append(f"En tête : **{top_lbl}** ({_fmt(top_val)}{unit_s}).")
        if len(labels) > 1:
            suite = ", ".join(f"{l} ({_fmt(v)})" for l, v in list(zip(labels, values))[1:3])
            parts.append(f"Suivi de {suite}.")
        try:
            total = sum(float(v) for v in values)
            parts.append(f"Total affiché ({len(labels)} éléments) : {_fmt(total)}{unit_s}.")
        except (TypeError, ValueError):
            pass
        return " ".join(p for p in parts if p)

    if ptype == "scalar":
        return f"**{title}** : {_fmt(payload.get('value'))}{unit_s}."

    if ptype == "table":
        cols = payload.get("columns", []) or []
        rows = payload.get("rows", []) or []
        head = f"**{title}** — " if title else ""
        return f"{head}{len(rows)} ligne(s), {len(cols)} colonne(s)."

    # type "raw" ou inconnu
    return str(payload.get("value", payload))[:500]


def parse_result(stdout: str):
    """Parse le stdout du code généré -> (payload_dict, summary_text).

    Robuste : tente le JSON complet, sinon la dernière ligne JSON ; en dernier
    recours emballe le texte brut dans un payload de type "raw".
    """
    text = (stdout or "").strip()
    payload = None
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            for line in reversed(text.splitlines()):
                line = line.strip()
                if line.startswith("{") or line.startswith("["):
                    try:
                        payload = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

    if payload is None:
        payload = {"type": "raw", "value": text}
    elif not (isinstance(payload, dict) and "type" in payload):
        # Sortie legacy (dict/scalaire sans "type") -> on emballe.
        payload = {"type": "raw", "value": payload}

    return payload, summarize_payload(payload)
# ────────────────────────────────────────────────────────────


def run_sandboxed(code: str):
    """Valide (AST) puis exécute le code dans un subprocess isolé avec timeout.

    Retourne (returncode, stdout, stderr). returncode == -1 pour un rejet
    sandbox ou un timeout (pas d'exécution réelle).
    """
    # Neutralise les `assert` défensifs (-> no-op) avant validation/exécution.
    code = sanitize_code(code)
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

    Réponse JSON :
      - result   : payload structuré (dict avec "type": bar|line|pie|scalar|table|raw)
      - summary  : synthèse textuelle FR du résultat
      - error    : message d'erreur si l'exécution a échoué
      - returncode : 0 si succès
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

    # Construction de la réponse : donnée structurée + synthèse textuelle.
    if returncode == 0:
        payload, summary = parse_result(stdout)
    else:
        payload, summary = None, ""

    # Enregistrer le résultat ou l'erreur à la racine de la trace
    if span:
        span.set_outputs({
            "result": payload,
            "summary": summary,
            "raw_stdout": stdout,
            "error": stderr,
            "success": returncode == 0,
        })

    return {
        'result': payload,
        'summary': summary,
        'error': stderr,
        'returncode': returncode,
    }

@mlflow.trace(name="vllm-generation", span_type="LLM")
def _call_llm(prompt: str) -> str:
    """Appelle le serveur LLM (API OpenAI-compatible) et renvoie le code extrait."""
    r = requests.post(LLM_URL,
                     headers={'Authorization': 'Bearer token', 'Content-Type': 'application/json'},
                     json={
        "model": LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": LLM_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=LLM_TIMEOUT)
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