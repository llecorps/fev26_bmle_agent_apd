from fastapi import FastAPI, Header
from pydantic import BaseModel
import requests, re
import tempfile
import subprocess
import sys
import os

from sandbox import validate_code, CodeValidationError


api = FastAPI(
    title="REST API",
    description="API powered by FastAPI.",
    version="0.0.1")

# Fichier de données exposé au code généré. Par défaut la sortie du pipeline DVC
# (data/processed/apd_clean.parquet) montée dans le conteneur sous /data.
DATA_PATH = os.getenv("DATA_PATH", "/data/processed/apd_clean.parquet")

# Endpoint du serveur LLM (vLLM/MLX), surchargeable hors Docker Desktop.
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


@api.post('/explore')
def post_explore(request: ChatRequest):
    """Returns explored data.
    """

    print(f"Requête reçue : {request.message}")

    code = generate_code_with_llm(request.message)
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
            code = repair_code_with_llm(request.message, code, stderr)

    return {'result': stdout, 'error': stderr, 'returncode': returncode}

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
    generated_code = extract_pure_code(response_dict['choices'][0]['message']['content'])
    print(generated_code)
    return generated_code


def generate_code_with_llm(chat_request: str) -> str:
    prompt = """[INST] Tu es un expert en analyse de données Python et Pandas. Ton unique tâche est de générer du code Python propre, optimisé et prêt à être exécuté dans un notebook Jupyter.
    CRITÈRES STRICTS :
    1. Ne retourne QUE le code Python à l'intérieur d'un seul bloc de code Markdown (```python ... ```).
    2. Ne saisis AUCUN texte d'introduction, AUCUNE explication, ni AUCUN commentaire après le code.
    3. Si tu as besoin d'expliquer quelque chose, fais-le uniquement sous forme de commentaires DICTÉS À L'INTÉRIEUR du code Python (ex: # Étape 1 : ...).

    Charger le fichier de données avec : df = pd.read_parquet('%(data_path)s')
    Le fichier contient des données nettoyées sur l'aide publique au développement française. Chaque ligne représente une déclaration de projet.

    Colonnes du DataFrame (utilise EXACTEMENT ces noms, accents compris) :
    %(schema)s

    Toujours utiliser Pandas pour effectuer des opérations sur les données, telles que le filtrage, l'agrégation et le regroupement.
    Commence TOUJOURS par les imports nécessaires : `import pandas as pd` et `import json`.
    Termine TOUJOURS par print(json.dumps(...)) pour imprimer le résultat final en JSON.

    N'importe AUCUN module autre que pandas, numpy et json.

    EXEMPLE pour « Montant moyen de X par catégorie Y » :
    ```python
    import pandas as pd
    import json
    df = pd.read_parquet('%(data_path)s')
    resultat = df.groupby('Y')['X'].mean()
    print(json.dumps(resultat.to_dict(), ensure_ascii=False))
    ```
    Adapte les noms de colonnes à la question. Pour un classement, utilise
    .sort_values(ascending=False).head(n). Pour un comptage, .value_counts().

    Format de réponse attendu :
    ```python
    # Ton code ici
    ``` [/INST]

    voici la requête de l'utilisateur :
    """ % {"data_path": DATA_PATH, "schema": SCHEMA_TEXT}
    prompt += chat_request

    print("Prompt envoyé au modèle :", prompt[:500], "...")
    return _call_llm(prompt)


def repair_code_with_llm(chat_request: str, broken_code: str, error: str) -> str:
    """Renvoie au LLM le code fautif et son erreur d'exécution pour correction."""
    prompt = """[INST] Le code Python suivant, censé répondre à la question d'un utilisateur, a échoué à l'exécution.

    Question : %(question)s

    Code fautif :
    ```python
    %(code)s
    ```

    Erreur :
    %(error)s

    Corrige le code. Règles : pas d'import autre que pandas/numpy/json, commence par les imports,
    termine par print(json.dumps(...)). Charge les données avec pd.read_parquet('%(data_path)s').
    Ne retourne QUE le code corrigé dans un bloc ```python ... ```. [/INST]
    """ % {"question": chat_request, "code": broken_code,
           "error": error.strip()[-800:], "data_path": DATA_PATH}
    print("Réparation demandée au modèle.")
    return _call_llm(prompt)


def extract_pure_code(llm_response: str) -> str:
    # Recherche le bloc de code ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Si le modèle a oublié les backticks mais a craché du code
    return llm_response.strip()
