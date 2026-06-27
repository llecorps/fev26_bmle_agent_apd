FROM ghcr.io/mlflow/mlflow:v3.14.0

# Installation des dépendances système nécessaires au healthcheck et à psycopg2
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installation du pilote Python
RUN pip install psycopg2-binary