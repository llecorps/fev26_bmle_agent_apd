FROM python:3.11-slim

WORKDIR /app

# Dépendances Python épinglées (pandas/numpy ont des wheels sur slim,
# pas besoin de build-essential).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY explore.py /app/explore.py
COPY sandbox.py /app/sandbox.py

# Point de montage du volume de données (monté en lecture seule via compose).
RUN mkdir /data

EXPOSE 8080

# démarrer l'API (on utilise l'hôte 0.0.0.0 pour qu'elle soit accessible de l'extérieur)
CMD ["uvicorn", "explore:api", "--host", "0.0.0.0", "--port", "8080"]
