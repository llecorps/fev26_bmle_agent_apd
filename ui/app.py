import streamlit as st
import requests
import os

# L'URL de l'API est définie via une variable d'environnement (pratique avec Docker)
# Par défaut, elle pointera vers le nom du service Docker "api" sur le port 8000
API_URL = os.getenv("API_URL", "http://explore-api:8080/explore")

st.title("💬 Chatbot APD")

# Zone de saisie pour l'utilisateur
if prompt := st.chat_input("Écrivez votre message ici..."):
    
    # Affichage du message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # Envoi à l'API et affichage de la réponse
    with st.chat_message("assistant"):
        with st.spinner("Le chatbot réfléchit... "):
            try:
                # L'appel à l'API bloque l'exécution, le spinner reste visible
                response = requests.post(API_URL, json={"message": prompt})
                response.raise_for_status()
                
                bot_reply = response.json().get("result", "Erreur : Réponse vide.")
                
                st.markdown(bot_reply)
                
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")
