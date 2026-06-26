"""
Dashboard APD — Dash/Plotly sur le port 8050.
Lit clean_data/apd_clean.csv mis à jour par le DAG Airflow.
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

CLEAN_DIR = os.getenv("CLEAN_DIR", "/app/clean_data")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/data")

app = Dash(__name__, title="APD France — Dashboard")
app.layout = html.Div([

    html.Div([
        html.H1("🌍 Aide Publique au Développement — France",
                style={"margin": "0", "color": "#1a1a2e"}),
        html.P("Mis à jour toutes les heures via Airflow",
               style={"margin": "4px 0 0", "color": "#666", "fontSize": "13px"}),
    ], style={"padding": "24px 32px 16px", "borderBottom": "2px solid #e8e8e8",
              "background": "#fafafa"}),

    html.Div([
        # ── Filtres ──────────────────────────────────────────────────────────
        html.Div([
            html.Label("Année", style={"fontWeight": "600"}),
            dcc.RangeSlider(id="year-slider", min=2010, max=2024, step=1,
                            value=[2019, 2024],
                            marks={y: str(y) for y in range(2010, 2025, 2)}),
            html.Br(),
            html.Label("Région", style={"fontWeight": "600"}),
            dcc.Dropdown(id="region-filter", multi=True,
                         placeholder="Toutes les régions"),
            html.Br(),
            html.Label("Secteur", style={"fontWeight": "600"}),
            dcc.Dropdown(id="sector-filter", multi=True,
                         placeholder="Tous les secteurs"),
        ], style={"width": "22%", "padding": "24px 16px",
                  "background": "#f5f5f5", "borderRight": "1px solid #e0e0e0",
                  "minHeight": "calc(100vh - 100px)"}),

        # ── Graphiques ───────────────────────────────────────────────────────
        html.Div([
            # KPIs
            html.Div(id="kpi-cards",
                     style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

            # Graphiques ligne 1
            html.Div([
                dcc.Graph(id="graph-top-pays", style={"flex": "1"}),
                dcc.Graph(id="graph-secteurs", style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

            # Graphiques ligne 2
            html.Div([
                dcc.Graph(id="graph-evolution", style={"flex": "2"}),
                dcc.Graph(id="graph-regions", style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

            # Modèle
            html.Div(id="model-card",
                     style={"background": "#e8f4fd", "borderRadius": "8px",
                            "padding": "16px", "marginTop": "8px"}),

        ], style={"flex": "1", "padding": "24px"}),

    ], style={"display": "flex"}),

    dcc.Interval(id="interval", interval=60_000, n_intervals=0),  # refresh 1 min
])


def load_data():
    path = os.path.join(CLEAN_DIR, "apd_clean.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_meta():
    path = os.path.join(MODEL_DIR, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Callback filtres ─────────────────────────────────────────────────────────
@app.callback(
    Output("region-filter", "options"),
    Output("sector-filter", "options"),
    Input("interval", "n_intervals"),
)
def update_filters(_):
    df = load_data()
    if df.empty:
        return [], []
    regions = sorted(df["Région"].dropna().unique())
    sectors = sorted(df["Secteur"].dropna().unique())
    return (
        [{"label": r, "value": r} for r in regions],
        [{"label": s, "value": s} for s in sectors],
    )


# ── Callback graphiques ───────────────────────────────────────────────────────
@app.callback(
    Output("kpi-cards",    "children"),
    Output("graph-top-pays",  "figure"),
    Output("graph-secteurs",  "figure"),
    Output("graph-evolution", "figure"),
    Output("graph-regions",   "figure"),
    Output("model-card",      "children"),
    Input("year-slider",   "value"),
    Input("region-filter", "value"),
    Input("sector-filter", "value"),
    Input("interval",      "n_intervals"),
)
def update_graphs(years, regions, sectors, _):
    df = load_data()
    if df.empty:
        empty = go.Figure()
        empty.update_layout(title="Données non disponibles")
        return [], empty, empty, empty, empty, "Aucune donnée"

    # Filtres
    df = df[(df["Annee de declaration"] >= years[0]) &
            (df["Annee de declaration"] <= years[1])]
    if regions:
        df = df[df["Région"].isin(regions)]
    if sectors:
        df = df[df["Secteur"].isin(sectors)]

    if df.empty:
        empty = go.Figure()
        return [], empty, empty, empty, empty, "Aucun résultat pour ces filtres"

    montant_col = "Montant verse (K EUR)"

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_m = df[montant_col].sum() / 1_000
    n_projets = len(df)
    n_pays = df["Pays beneficiaire"].nunique()
    ticket_moy = df[montant_col].mean()

    def kpi_card(label, value):
        return html.Div([
            html.P(label, style={"margin": "0", "fontSize": "12px", "color": "#666"}),
            html.H3(value, style={"margin": "4px 0 0", "color": "#1a1a2e"}),
        ], style={"background": "white", "borderRadius": "8px", "padding": "16px",
                  "flex": "1", "boxShadow": "0 1px 4px rgba(0,0,0,.08)"})

    kpis = [
        kpi_card("Montant total versé", f"{total_m:,.0f} M EUR"),
        kpi_card("Projets",             f"{n_projets:,}"),
        kpi_card("Pays bénéficiaires",  str(n_pays)),
        kpi_card("Ticket moyen",        f"{ticket_moy:,.0f} K EUR"),
    ]

    # ── Top 10 pays ───────────────────────────────────────────────────────────
    top_pays = (df.groupby("Pays beneficiaire")[montant_col]
                  .sum().nlargest(10).reset_index())
    fig_pays = px.bar(top_pays, x=montant_col, y="Pays beneficiaire",
                      orientation="h", title="Top 10 pays bénéficiaires",
                      labels={montant_col: "Montant versé (K EUR)", "Pays beneficiaire": ""},
                      color=montant_col, color_continuous_scale="Blues")
    fig_pays.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))

    # ── Répartition par secteur ───────────────────────────────────────────────
    secteurs = (df.groupby("Secteur")[montant_col]
                  .sum().nlargest(8).reset_index())
    fig_sect = px.pie(secteurs, values=montant_col, names="Secteur",
                      title="Répartition par secteur (top 8)",
                      hole=0.4)
    fig_sect.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # ── Évolution annuelle ────────────────────────────────────────────────────
    evol = (df.groupby("Annee de declaration")[montant_col]
              .sum().reset_index())
    fig_evol = px.area(evol, x="Annee de declaration", y=montant_col,
                       title="Évolution annuelle des montants versés",
                       labels={montant_col: "Montant versé (K EUR)",
                               "Annee de declaration": "Année"})
    fig_evol.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    # ── Répartition par région ────────────────────────────────────────────────
    regions_df = (df.groupby("Région")[montant_col].sum().reset_index())
    fig_reg = px.bar(regions_df, x="Région", y=montant_col,
                     title="Montants par région",
                     labels={montant_col: "K EUR", "Région": ""},
                     color=montant_col, color_continuous_scale="Greens")
    fig_reg.update_layout(coloraxis_showscale=False,
                          xaxis_tickangle=-30, margin=dict(l=0, r=0, t=40, b=60))

    # ── Carte modèle ─────────────────────────────────────────────────────────
    meta = load_meta()
    if meta:
        model_card = [
            html.H4(f"🤖 Modèle actif : {meta['model']}", style={"margin": "0 0 8px"}),
            html.Div([
                html.Span(f"R² = {meta['r2']:.3f}", style={"marginRight": "24px"}),
                html.Span(f"RMSE = {meta['rmse']:.3f}", style={"marginRight": "24px"}),
                html.Span(f"MAE = {meta['mae']:.3f}"),
            ], style={"fontSize": "14px", "color": "#444"}),
            html.P(f"Entraîné sur {meta['train_size']:,} projets "
                   f"| testé sur {meta['test_size']:,} projets",
                   style={"margin": "8px 0 0", "fontSize": "12px", "color": "#777"}),
        ]
    else:
        model_card = [html.P("Modèle non disponible — en attente du premier run Airflow.")]

    return kpis, fig_pays, fig_sect, fig_evol, fig_reg, model_card


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
