import streamlit as st
import requests
import os

import pandas as pd
import plotly.express as px

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
# RENDU D'UN RÉSULTAT (texte + graphique)
# ═════════════════════════════════════════════════
def render_payload(payload: dict, summary: str, key: str):
    """Affiche la synthèse textuelle puis le graphique adapté au type."""
    if summary:
        st.markdown(summary)

    if not isinstance(payload, dict):
        st.write(payload)
        return

    ptype = payload.get("type")
    title = payload.get("title", "")
    unit = payload.get("unit", "") or ""

    if ptype in ("bar", "line", "pie"):
        labels = payload.get("labels", []) or []
        values = payload.get("values", []) or []
        if not labels or not values:
            st.info("Aucune donnée à représenter.")
            return

        df = pd.DataFrame({
            payload.get("x_label", "Catégorie"): labels,
            payload.get("y_label", "Valeur"): values,
        })
        xcol, ycol = df.columns[0], df.columns[1]
        y_title = f"{ycol} ({unit})" if unit else ycol

        if ptype == "bar":
            fig = px.bar(df, x=xcol, y=ycol, title=title, text_auto=".2s")
            fig.update_layout(xaxis_title=xcol, yaxis_title=y_title)
            fig.update_xaxes(categoryorder="total descending")
        elif ptype == "line":
            fig = px.line(df, x=xcol, y=ycol, title=title, markers=True)
            fig.update_layout(xaxis_title=xcol, yaxis_title=y_title)
        else:  # pie
            fig = px.pie(df, names=xcol, values=ycol, title=title, hole=0.35)

        st.plotly_chart(fig, use_container_width=True, key=key)
        with st.expander("Voir les données"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif ptype == "scalar":
        value = payload.get("value")
        label = f"{title} ({unit})" if unit else title
        try:
            st.metric(label or "Résultat", f"{float(value):,.2f}".replace(",", " "))
        except (TypeError, ValueError):
            st.metric(label or "Résultat", str(value))

    elif ptype == "table":
        cols = payload.get("columns", [])
        rows = payload.get("rows", [])
        if title:
            st.markdown(f"**{title}**")
        try:
            df = pd.DataFrame(rows, columns=cols if cols else None)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.json({"columns": cols, "rows": rows})

    else:  # "raw" ou inconnu
        value = payload.get("value", payload)
        if isinstance(value, (dict, list)):
            st.json(value)
        else:
            st.write(value)


# ═════════════════════════════════════════════════
# PAGE : CHATBOT EXPLORATION
# ═════════════════════════════════════════════════
if page == "Chatbot Exploration":
    st.title("💬 Chatbot Exploration")
    st.caption("Posez une question en langage naturel — la réponse s'affiche en texte et en graphique.")

    # Historique de conversation persistant
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Ré-affichage de l'historique
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif msg.get("error"):
                st.error(msg["error"])
            else:
                render_payload(msg.get("payload"), msg.get("summary", ""), key=f"hist_{i}")

    if prompt := st.chat_input("Écrivez votre message ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Le chatbot réfléchit... "):
                try:
                    response = requests.post(EXPLORE_API_URL, json={"message": prompt}, timeout=180)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.RequestException as e:
                    err = f"Erreur de connexion à l'API : {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "error": err})
                else:
                    if data.get("returncode") == 0 and data.get("result") is not None:
                        payload = data["result"]
                        summary = data.get("summary", "")
                        idx = len(st.session_state.messages)
                        render_payload(payload, summary, key=f"live_{idx}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "payload": payload,
                            "summary": summary,
                        })
                    else:
                        err = data.get("error") or "Le chatbot n'a pas pu produire de résultat."
                        st.error("Désolé, je n'ai pas réussi à répondre.")
                        with st.expander("Détail de l'erreur"):
                            st.code(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "error": "Le chatbot n'a pas réussi à répondre à cette question.",
                        })


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