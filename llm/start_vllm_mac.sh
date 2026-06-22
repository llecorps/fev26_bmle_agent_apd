#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
VENV_DIR="$HOME/.venv-vllm-mlx"
MODEL="mlx-community/Mistral-7B-Instruct-v0.3-4bit"
PORT=8000

# Colle ton jeton Hugging Face (hf_...) entre les guillemets ci-dessous si nécessaire
HF_TOKEN="<HUGGING_FACE_TOKEN>"
# ==========================================

echo "======================================================"
echo "  Configuration du serveur vLLM (MLX) sur Apple Silicon"
echo "======================================================"

# Sécurité : Vérifier qu'on est bien sur un Mac Apple Silicon (M1/M2/M3/M4)
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "/!\ Erreur : Ce script nécessite l'architecture Apple Silicon (arm64)."
    exit 1
fi
echo " Processeur Apple Silicon détecté."

# Authentification Hugging Face (si le token est renseigné)
if [ -n "$HF_TOKEN" ]; then
    echo " Jeton Hugging Face configuré."
    export HF_TOKEN="$HF_TOKEN"
else
    echo "  Aucun HF_TOKEN fourni (généralement OK pour les modèles publics comme Mistral)."
fi

# Configuration cruciale pour l'architecture Apple (MPS)
echo "  Configuration du périphérique cible sur 'mps'..."
export VLLM_TARGET_DEVICE="mps"

# Vérifier que les outils de compilation Xcode sont installés
if ! xcode-select -p &>/dev/null; then
    echo "⏳ Outils de ligne de commande Xcode manquants. Lancement de l'installation..."
    xcode-select --install
    echo "  Veuillez terminer l'installation de l'invite Apple qui vient de s'ouvrir, puis relancez ce script."
    exit 1
fi
echo " Outils Xcode détectés."

# 5. Création et activation de l'environnement virtuel Python
if [ ! -d "$VENV_DIR" ]; then
    echo " Création d'un environnement virtuel Python isolé..."
    python3 -m venv "$VENV_DIR"
fi

echo " Activation de l'environnement virtuel..."
source "$VENV_DIR/bin/activate"

# Mise à jour des outils de paquets
pip install --upgrade pip setuptools wheel

# Installation du moteur d'inférence vllm-mlx
if ! command -v vllm-mlx &> /dev/null; then
    echo " Installation de vllm-mlx (Moteur vLLM optimisé Metal)..."
    pip install vllm-mlx
else
    echo " vllm-mlx est déjà installé."
fi

# 7. Lancement du serveur d'API compatible OpenAI
echo "======================================================"
echo " DÉMARRAGE DU SERVEUR"
echo " Modèle : $MODEL"
echo " Port   : $PORT"
echo " Device : $VLLM_TARGET_DEVICE"
echo "------------------------------------------------------"
echo " Note : Au premier lancement, le script va télécharger"
echo "          les ~4.5 Go du modèle. Patience !"
echo "======================================================"

vllm-mlx serve "$MODEL" --port "$PORT" --continuous-batching
