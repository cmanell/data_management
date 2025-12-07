from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import json
from streamlit_plotly_events import plotly_events
import numpy as np


# =============================
# 0. Chargement des données
# =============================
df_tranch_age = pd.read_csv("df_tranch_age.csv", sep=",")
df_tot_age    = pd.read_csv("df_tot_age.csv",    sep=",")
df_sejour     = pd.read_csv("df_sejour.csv",     sep=",")

# GeoJSON pour la carte
with open("departements.geojson", encoding="utf-8") as f:
    dep_geojson = json.load(f)


# =============================
# 1. Configuration de la page
# =============================
st.set_page_config(
    page_title="Projet Data : Morbidités hospitalières",
    layout="wide"
)

st.title("Projet Data : Morbidité hospitalière en France métropolitaine + Corse")
st.markdown("### Taux de recours aux établissements de santé")
st.markdown("#### 1. Vue d’ensemble par département et évolution temporelle")
st.markdown("---")


# =============================
# 2. Carte interactive par pathologie
# =============================
st.subheader("1️⃣ Carte interactive : Taux de recours par département")

# Liste déroulante des pathologies (sans doublons)
pathologies_tout = sorted(df_tot_age["Pathologie"].drop_duplicates())

pathologie_selected = st.selectbox(
    "Pathologie analysée :",
    pathologies_tout
)

selected_points = []  # contiendra le département cliqué

if pathologie_selected:

    # Filtre du DF principal
    df_p = df_tot_age[df_tot_age["Pathologie"] == pathologie_selected].copy()

    fig_carte = px.choropleth(
        df_p,
        geojson=dep_geojson,
        locations="dep_code",
        featureidkey="properties.code",
        color="nbr recours",
        animation_frame="ANNEE",
        color_continuous_scale="Blues",
        range_color=(0, df_p["nbr recours"].max()),
        labels={"nbr recours": "Taux de recours"},
        hover_name="Département",
        hover_data={"Pathologie": True, "Département": False}
    )

    fig_carte.update_geos(fitbounds="locations", visible=False)

    fig_carte.update_layout(
        title=f"Évolution du Taux de recours pour {pathologie_selected} (H+F) par département",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    # Récupérer le département cliqué
    selected_points = plotly_events(
        fig_carte,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=500,
        override_width="100%"
    )
else:
    st.info("Veuillez sélectionner une pathologie pour afficher la carte.")


# =============================
# 3. Contruction des graphiques détaillés
# =============================
col1, col2 = st.columns(2)

# ---------------------------------
# 3.1 Durée des séjours + gaussienne
# ---------------------------------
with col1:
    st.subheader("2️⃣ Durée des séjours pour la pathologie sélectionnée")

    if selected_points:
        # Année sélectionnée
        annees = sorted(df_tot_age["ANNEE"].drop_duplicates())
        annee_selected = st.selectbox("Année :", annees, index=len(annees) - 1)

        # Département sélectionné sur la carte
        idx = selected_points[0]["pointNumber"]
        dep_code = df_p.iloc[idx]["dep_code"]
        st.markdown(f"**Département sélectionné :** `{dep_code}`")

        st.markdown(
            f"Distribution du nombre de séjours pour **{pathologie_selected}** "
            f"dans le département **{dep_code}** en **{annee_selected}**."
        )

        # Filtre des données de séjours
        df_sejour_filt = df_sejour[
            (df_sejour["dep_code"] == dep_code)
            & (df_sejour["Pathologie"] == pathologie_selected)
            & (df_sejour["ANNEE"] == annee_selected)
        ].copy()

        if not df_sejour_filt.empty:
            # Histogramme des durées de séjour
            fig_sejour = px.bar(
                df_sejour_filt,
                y="Nombre séjours",
                x="Durée séjour",
                color="Durée séjour",
                title="Nombre de séjours par plage de durée d’hospitalisation",
                labels={
                    "Nombre séjours": "Nombre de séjours",
                    "Durée séjour": "Durée du séjour (en jours)"
                },
                barmode="group",
                text="ratio durée du séjour",
            )
            y_range_sejour = df_sejour_filt["Nombre séjours"].max() * 1.1

            fig_sejour.update_traces(
            texttemplate='%{text:.2f}%', 
            textposition='outside')

            fig_sejour.update_traces(width=0.8)  
            fig_sejour.update_layout(bargap=1, legend_title_text="Plage de durée")
            fig_sejour.update_yaxes(range=[0, y_range_sejour])

            st.plotly_chart(fig_sejour, use_container_width=True)

            # Distribution normale théorique
            x = df_sejour_filt["Durée_num"]
            w = df_sejour_filt["Nombre séjours"]

            mu = np.average(x, weights=w)
            sigma = np.sqrt(np.average((x - mu) ** 2, weights=w))

            x_curve = np.linspace(min(x), max(x), 300)
            y_curve = norm.pdf(x_curve, mu, sigma)
            y_curve = y_curve * w.sum() / y_curve.sum()

            fig_gauss = go.Figure()
            fig_gauss.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode="lines",
                    name="Courbe normale théorique",
                    line=dict(color="white", width=3),
                )
            )

            fig_gauss.add_vline(x=mu, line_dash="dash", line_color="red", name="Moyenne")
            fig_gauss.add_vline(x=mu + sigma, line_dash="dot", line_color="blue")

            fig_gauss.add_annotation(
                x=mu,
                y=max(y_curve) * 0.95,
                text=f"µ = {mu:.2f} j",
                showarrow=False,
                font=dict(color="red", size=14),
            )

            fig_gauss.add_annotation(
                x=mu + sigma,
                y=max(y_curve) * 0.85,
                text=f"µ + σ = {mu + sigma:.2f} j",
                showarrow=False,
                font=dict(color="blue"),
            )

            fig_gauss.update_layout(
                title="Distribution normale théorique de la durée de séjour",
                xaxis_title="Durée du séjour (jours)",
                yaxis_title="Nombre théorique de séjours (normalisé)",
                template="plotly_white",
            )

            st.plotly_chart(fig_gauss, use_container_width=True)

        else:
            st.info("Pas de données de durée de séjour disponibles pour ce département / cette année.")
    else:
        st.info("Cliquez sur un département de la carte pour afficher les détails.")


# ---------------------------------
# 3.2 Répartition par sexe + 3.3 par tranche d’âge
# ---------------------------------
with col2:
    st.subheader("3️⃣ Répartition des séjours selon le sexe")

    if selected_points:
        # On réutilise dep_code et annee_selected définis dans col1
        idx = selected_points[0]["pointNumber"]
        dep_code = df_p.iloc[idx]["dep_code"]

        st.markdown(f"**Département sélectionné :** `{dep_code}`")

        if "annee_selected" in locals():
            st.markdown(
                f"Taux de recours pour **{pathologie_selected}** "
                f"dans le département **{dep_code}** en **{annee_selected}**, par sexe."
            )

            df_tot_age_filt = df_tot_age[
                (df_tot_age["dep_code"] == dep_code)
                & (df_tot_age["Pathologie"] == pathologie_selected)
                & (df_tot_age["ANNEE"] == annee_selected)
            ].copy()

            if not df_tot_age_filt.empty:
                fig_sexe = px.bar(
                    df_tot_age_filt,
                    y="nbr recours",
                    x="SEXE",
                    color="SEXE",
                    title="Distribution du Taux de recours selon le sexe",
                    labels={
                        "nbr recours": "Taux de recours",
                        "SEXE": "Sexe",
                    },
                    barmode="group",
                    text="ratio par sexe",
                    color_discrete_map={
                        "Homme": "#318CE7",
                        "Femme": "#DE3163",
                    },
                )
                y_range1 = df_tot_age_filt["nbr recours"].max() * 1.1

                fig_sexe.update_traces(
                texttemplate='%{text:.2f}%', 
                textposition='outside')
                fig_sexe.update_traces(width=0.2)  
                fig_sexe.update_yaxes(range=[0, y_range1])

                st.plotly_chart(fig_sexe, use_container_width=True)
            else:
                st.info("Pas de données de répartition par sexe pour ce département / cette année.")
        else:
            st.info("Veuillez d’abord choisir une année dans la colonne de gauche.")
    else:
        st.info("Cliquez sur un département sur la carte pour voir la répartition par sexe.")

    st.markdown("---")
    st.subheader("4️⃣ Répartition des séjours par tranche d’âge")

    if selected_points and "annee_selected" in locals():
        df_tranch_filt = df_tranch_age[
            (df_tranch_age["dep_code"] == dep_code)
            & (df_tranch_age["Pathologie"] == pathologie_selected)
            & (df_tranch_age["ANNEE"] == annee_selected)
        ].copy()

        if not df_tranch_filt.empty:
            fig_age = px.bar(
                df_tranch_filt,
                y="nbr recours",
                x="Tranche d'age",
                color="Tranche d'age",
                title="Distribution du Taux de recours par tranche d’âge",
                labels={
                    "nbr recours": "Taux de recours",
                    "Tranche d'age": "Tranche d’âge",
                },
                barmode="group",
                text="ratio par tranche d'age",
            )
            y_range2 = df_tranch_filt["nbr recours"].max() * 1.1

            fig_age.update_traces(
            texttemplate='%{text:.2f}%', 
            textposition='outside')     
            fig_age.update_traces(width=0.6)  
            fig_age.update_layout(legend_title_text="Tranche d’âge")
            fig_age.update_yaxes(range=[0, y_range2])

            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Pas de données par tranche d’âge pour ce département / cette année.")
    elif not selected_points:
        st.info("Cliquez sur un département sur la carte pour voir la répartition par tranche d’âge.")
