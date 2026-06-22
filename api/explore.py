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


@api.post('/explore')
def post_explore(request: ChatRequest):
    """Returns explored data.
    """

    print(f"Requête reçue : {request.message}")

    # Generate code using LLM (Mistral-7B) based on the chat request
    gen_code = generate_code_with_llm(request.message)

    # Validation AST AVANT toute exécution : rejette le code dangereux
    # (imports hors whitelist, eval/exec/open, écritures fichier, ...).
    try:
        validate_code(gen_code)
    except CodeValidationError as exc:
        print(f"Code rejeté par la sandbox : {exc}")
        return {'result': '', 'error': f"Code rejeté par la sandbox : {exc}", 'returncode': -1}

    # Create a temporary file and write the generated code to it
    with tempfile.NamedTemporaryFile(delete=True, suffix='.py', mode='w+t') as temp_file:
        # Write data to the file
        temp_file.write(gen_code)
        temp_file.flush()  # Ensure data is written to disk

        # Go back to the beginning of the file to read it
        temp_file.seek(0)
        print(f"Temporary file created at: {temp_file.name}")
        #print(f"Content: {temp_file.read()}")

        print(f"Exécution locale du fichier temporaire : {temp_file.name}\n")
        # `-I` : mode isolé (ignore variables d'env PYTHON* et user site-packages).
        # `timeout` : tue le process si le code boucle ou traîne.
        try:
            r = subprocess.run([sys.executable, "-I", temp_file.name],
                               capture_output=True,
                               text=True,
                               timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"Exécution interrompue après {EXEC_TIMEOUT:.0f}s (timeout)")
            return {'result': '', 'error': f"Exécution interrompue après {EXEC_TIMEOUT:.0f}s (timeout)", 'returncode': -1}

    # Once the 'with' block ends, the file is automatically deleted from the disk!

    # Affichage des résultats
    print("\n--- Résultat de l'exécution ---")
    if r.returncode == 0:
        print(r.stdout)
    else:
        print("Erreur lors de l'exécution du code Pandas :")
        print(r.stderr)


    return {'result': r.stdout, 'error': r.stderr, 'returncode': r.returncode}

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
    Toujours imprimer le résultat final sous forme de JSON avec print(...).

    N'importe AUCUN module autre que pandas, numpy et json.

    Format de réponse attendu :
    ```python
    # Ton code ici
    ``` [/INST]

    voici la requête de l'utilisateur :
    """ % {"data_path": DATA_PATH, "schema": SCHEMA_TEXT}
    prompt += chat_request

    print("Prompt envoyé au modèle :", prompt)

    # creating a POST request
    r = requests.post(LLM_URL,
                     headers={'Authorization': 'Bearer token', 'Content-Type': 'application/json'},
                     json={
        "model": LLM_MODEL,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }, timeout=120)

    # getting the response elements
    response_dict = r.json()

    print("Response Header:", r.headers)
    #fix ->#print("Status Code:", r.headers['status'])
    #print("Response Body:", response_dict)
    generated_code = extract_pure_code(response_dict['choices'][0]['message']['content'])
    print(generated_code)

    return generated_code


def extract_pure_code(llm_response: str) -> str:
    # Recherche le bloc de code ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Si le modèle a oublié les backticks mais a craché du code
    return llm_response.strip()
