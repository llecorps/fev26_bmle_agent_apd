from fastapi import FastAPI, Header
import requests, re
import tempfile
import subprocess
import sys


api = FastAPI(
    title="REST API",
    description="API powered by FastAPI.",
    version="0.0.1")

@api.post('/explore')
def post_explore():
    """Returns explored data.
    """

    #Generate code using LLM (Mistral-7B)
    gen_code = generate_code_with_llm()

    
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
        r = subprocess.run([sys.executable, temp_file.name], 
                           capture_output=True, 
                           text=True)
        
    # Once the 'with' block ends, the file is automatically deleted from the disk!

    # Affichage des résultats
    print("\n--- Résultat de l'exécution ---")
    if r.returncode == 0:
        print(r.stdout)
        print(r)
    else:
        print("Erreur lors de l'exécution du code Pandas :")
        print(r.stderr)

    

    return {'Inference': 'Done'}

def generate_code_with_llm():
    prompt = """[INST] Tu es un expert en analyse de données Python et Pandas. Ton unique tâche est de générer du code Python propre, optimisé et prêt à être exécuté dans un notebook Jupyter.
    CRITÈRES STRICTS :
    1. Ne retourne QUE le code Python à l'intérieur d'un seul bloc de code Markdown (```python ... ```).
    2. Ne saisis AUCUN texte d'introduction, AUCUNE explication, ni AUCUN commentaire après le code.
    3. Si tu as besoin d'expliquer quelque chose, fais-le uniquement sous forme de commentaires DICTÉS À L'INTÉRIEUR du code Python (ex: # Étape 1 : ...).
    Voici ce que le code doit faire :
    Charger un fichier '/data/aide-publique-au-developpement_clean.csv' (avec comme paramètre encoding='utf-8', sep=';'), afficher les 10 premières lignes du DataFrame.]
    Format de réponse attendu :
    ```python
    # Ton code ici
    ``` [/INST]"""

    # creating a POST request
    r = requests.post('http://host.docker.internal:8000/v1/chat/completions', 
                     headers={'Authorization': 'Bearer token', 'Content-Type': 'application/json'}, 
                     json={
        "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

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
