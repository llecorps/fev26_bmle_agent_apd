FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires (si besoin de compiler certains packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    pandas


COPY explore.py /app/explore.py

# Créer un dossier pour accueillir le fichier CSV de données
RUN mkdir /data

EXPOSE 8080

# démarrer l'API (on utilise l'hôte 0.0.0.0 pour qu'elle soit accessible de l'extérieur)
CMD ["uvicorn", "explore:api", "--host", "0.0.0.0", "--port", "8080"]
