import streamlit as st
import requests
import os

# ─── URLs des API (configurables via Docker) ─────
EXPLORE_API_URL = os.getenv("EXPLORE_API_URL", "http://explore-api:8080/explore")
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://predict-api:8080")

# ─── Sidebar ─────────────────────────────────────
st.sidebar.title("🌍 APD France")
st.sidebar.markdown("**Aide Publique au Développement**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Chatbot Exploration",
    "Prédictions"
])

st.sidebar.markdown("---")
st.sidebar.caption("DataScientest — ML Engineer")


# ═════════════════════════════════════════════════
# PAGE : CHATBOT EXPLORATION
# ═════════════════════════════════════════════════
if page == "Chatbot Exploration":
    st.title("💬 Chatbot Exploration")

    if prompt := st.chat_input("Écrivez votre message ici..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Le chatbot réfléchit... "):
                try:
                    response = requests.post(EXPLORE_API_URL, json={"message": prompt})
                    response.raise_for_status()
                    bot_reply = response.json().get("result", "Erreur : Réponse vide.")
                    st.markdown(bot_reply)
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de connexion à l'API : {e}")


# ═════════════════════════════════════════════════
# PAGE : PRÉDICTIONS (via API /predict)
# ═════════════════════════════════════════════════
if page == "Prédictions":
    st.title("📊 Prédictions")
    st.markdown("Renseignez les caractéristiques d'un projet pour obtenir une estimation du montant engagé.")
    st.markdown("---")

    # ─── Charger les features depuis l'API ────────
    @st.cache_data(ttl=300)
    def load_features():
        """Récupère la liste des features et les valeurs possibles depuis l'API."""
        try:
            resp = requests.get(f"{PREDICT_API_URL}/model/features", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"Impossible de contacter l'API predict : {e}")
            return None

    features_info = load_features()

    if features_info is None:
        st.stop()

    dropdowns = features_info.get("dropdowns", {})
    cat_low = features_info.get("categorical_low", [])
    cat_high = features_info.get("categorical_high", [])

    # ─── Formulaire ───────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Caractéristiques du projet")

        col1, col2, col3 = st.columns(3)
        inputs = {}

        key_cats = ['Agence', 'Type de financement', 'Pays beneficiaire', 'Région',
                    'Secteur', 'Catégorie CAD', 'Bi/Multi.1']

        with col1:
            for c in key_cats[:3]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col2:
            for c in key_cats[3:6]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)
        with col3:
            for c in key_cats[6:]:
                if c in dropdowns:
                    inputs[c] = st.selectbox(c, dropdowns[c], index=0)

        with st.expander("➕ Paramètres avancés (optionnel)", expanded=False):
            c1, c2, c3 = st.columns(3)
            advanced_cats = [c for c in cat_low + cat_high if c not in key_cats and c in dropdowns]
            for i, c in enumerate(advanced_cats[:12]):
                with [c1, c2, c3][i % 3]:
                    inputs[c] = st.selectbox(c, ["(par défaut)"] + dropdowns.get(c, []), index=0)

        submitted = st.form_submit_button("🚀 Prédire le montant", type="primary", use_container_width=True)

    # ─── Appel API et affichage ───────────────────
    if submitted:
        # Ne garder que les features réellement renseignées
        features = {k: v for k, v in inputs.items() if v != "(par défaut)"}

        with st.spinner("Prédiction en cours..."):
            try:
                resp = requests.post(
                    f"{PREDICT_API_URL}/predict",
                    json={"features": features},
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")
                st.stop()

        st.markdown("---")
        st.subheader("Résultat de la prédiction")

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 log(1+engagement)", f"{data['log_prediction']:.2f}")
        c2.metric("💰 Montant estimé", data['montant_label'])

        montant_eur = data['montant_keur'] * 1000
        c3.metric("💶 En euros", f"{montant_eur:,.0f} €")

        # Fourchette
        low = data['fourchette_keur']['low'] * 1000
        high = data['fourchette_keur']['high'] * 1000
        st.info(f"📐 **Fourchette ±1 RMSE** : de {low:,.0f} € à {high:,.0f} € "
                f"(facteur ×3,7 sur le montant brut)")

        # Tranche
        tranche = data['tranche']
        if "Petit" in tranche:
            st.success(f"**Tranche estimée** : 🟢 {tranche}")
        elif "Moyen" in tranche:
            st.success(f"**Tranche estimée** : 🟡 {tranche}")
        elif "Grand" in tranche and "Très" not in tranche:
            st.success(f"**Tranche estimée** : 🟠 {tranche}")
        else:
            st.success(f"**Tranche estimée** : 🔴 {tranche}")