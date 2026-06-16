# 1. Utiliser une image Python officielle légère
FROM python:3.11-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires (si besoin de compiler certains packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Installer les dépendances Python nécessaires à votre script
# Note : j'inclus explicitement fastapi, uvicorn, requests, et pandas (car le code généré l'utilise)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    pandas

# 5. Copier le code de l'application dans le conteneur
COPY explore.py /app/explore.py

# 6. Créer un dossier pour accueillir le fichier CSV de données
RUN mkdir /data

# 7. Exposer le port sur lequel FastAPI va tourner
EXPOSE 8080

# 8. Commande pour démarrer l'API (on utilise l'hôte 0.0.0.0 pour qu'elle soit accessible de l'extérieur)
CMD ["uvicorn", "explore:api", "--host", "0.0.0.0", "--port", "8080"]
