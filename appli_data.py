import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import json
from scipy.stats import chi2_contingency, norm
import numpy as np

# =============================
# 0. Data loading with caching
# =============================
st.set_page_config(page_title="Projet Data : Morbidités hospitalières", layout="wide")

@st.cache_data
def load_data():
    df_tranch_age = pd.read_csv("df_tranch_age.csv", sep=",")
    df_tot_age    = pd.read_csv("df_tot_age.csv", sep=",")
    df_sejour     = pd.read_csv("df_sejour.csv", sep=",")
    df_tableau_1  = pd.read_csv("tableau_1.csv", sep=";")
    with open("departements.geojson", encoding="utf-8") as f:
        dep_geojson = json.load(f)
    return df_tranch_age, df_tot_age, df_sejour, df_tableau_1, dep_geojson

df_tranch_age, df_tot_age, df_sejour, df_tableau_1, dep_geojson = load_data()

# =============================
# 1. Page config
# =============================

st.title("Projet Data : Morbidité hospitalière en France métropolitaine + Corse")
st.markdown("### Taux de recours aux établissements de santé")
st.markdown("---")

# =============================
# 2. Sidebar filter
# =============================

st.sidebar.header("Filtres")
pathologies = sorted(df_tot_age['Pathologie'].drop_duplicates())
pathologie_selected = st.sidebar.selectbox("Pathologie :", pathologies, index=0)

# Department selector in sidebar
df_map_temp = df_tot_age[
    (df_tot_age["Pathologie"] == pathologie_selected)
].copy()
departments = sorted(df_map_temp['Département'].unique())
selected_dept = st.sidebar.selectbox(
    "Département :",
    options=["-- Aucun --"] + departments,
    key="dept_selector"
)

years = sorted(df_tot_age['ANNEE'].drop_duplicates())
year_selected = st.sidebar.selectbox("Année :", years, index=len(years)-1)

# =============================
# 3. Main layout
# =============================

st.header("1️⃣ Vue d'ensemble sur la carte de France")

# Prepare data for map
df_map = df_tot_age[
    (df_tot_age["Pathologie"] == pathologie_selected)
].copy()

# -----------------------------
# Top: Choropleth map (full width)
# -----------------------------

st.subheader("Carte interactive")

fig_map = px.choropleth(
    df_map,
    geojson=dep_geojson,
    locations="dep_code",
    animation_frame="ANNEE",
    featureidkey="properties.code",
    color="total cas",
    color_continuous_scale="Blues",
    range_color=(0, df_map["total cas"].max()),
    labels={"total cas": "Taux de recours"},
    hover_name="Département",
    hover_data={"Pathologie": True, "Département": False}
)

fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(
    title=f"Représentation du taux de recours par département - {pathologie_selected}",
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    height=700
)

st.plotly_chart(fig_map, use_container_width=True, key="choropleth_map")

# -----------------------------
# Bottom: Sex and Age charts side by side
# -----------------------------

if selected_dept and selected_dept != "-- Aucun --":
    # Get department info
    dept_info = df_map[df_map["Département"] == selected_dept].iloc[0]
    dep_code = dept_info["dep_code"]
    dep_name = dept_info["Département"]

    st.header(f"Détails pour le département : {dep_name} ({dep_code})")
    st.markdown(f"**Département sélectionné :** {dep_name} (`{dep_code}`)")
    st.markdown(f"**Pathologie sélectionnée :** {pathologie_selected}")
    # Filter the data (ONLY ONCE)
    df_tot_age_filt = df_tot_age[
        (df_tot_age["dep_code"] == dep_code) &
        (df_tot_age["Pathologie"] == pathologie_selected) &
        (df_tot_age["ANNEE"] == year_selected)
    ]
    
    # Add total case count
    total_cases = df_tot_age_filt["nbr recours"].sum()
    st.metric("Nombre total de cas", f"{total_cases:,.0f}")

    # Create two columns for sex and age charts
    col1, col2 = st.columns(2)
    
    # Sex distribution
    with col1:
        if not df_tot_age_filt.empty:

            
            fig_sex = px.bar(
                df_tot_age_filt,
                x='SEXE',
                y='nbr recours',
                color='SEXE', text='ratio par sexe',
                color_discrete_map={"Homme": "#318CE7", "Femme": "#DE3163"},
                title="Répartition par sexe sur toute la France"
            )
            y_range_sejour = df_tot_age_filt["nbr recours"].max() * 1.2

            fig_sex.update_traces(texttemplate='%{text:.1f}%', textposition='outside', width=0.3)
            fig_sex.update_layout(showlegend=False)
            fig_sex.update_yaxes(title_text='nbr recours %', range=[0, y_range_sejour])
            st.plotly_chart(fig_sex, use_container_width=True)

    # Age distribution
    with col2:
        df_tranch_filt = df_tranch_age[
            (df_tranch_age["dep_code"] == dep_code) &
            (df_tranch_age["Pathologie"] == pathologie_selected) &
            (df_tranch_age["ANNEE"] == year_selected)
        ]
        
        if not df_tranch_filt.empty:
            

            fig_age = px.bar(
                df_tranch_filt, 
                x="Tranche d'age", 
                y='nbr recours',
                color="Tranche d'age", 
                text='ratio par tranche d\'age',
                barmode='group', 
                title="Répartition par tranche d'âge"
            )

            fig_age.update_yaxes(
                title_text='nbr recours %',
                range=[0, df_tranch_filt['nbr recours'].max() * 1.15]
                )
            
            fig_age.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside', width=1
                )
            
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Pas de données détaillées pour ce département.")
else:
    st.info("👈 Sélectionnez un département dans la barre latérale pour voir les détails.")


# =============================
# 4. Duration of stay section
# =============================
if selected_dept and selected_dept != "-- Aucun --":
    st.markdown("---")

    st.header("Analyse de la durée du séjour")
    st.markdown("Histogramme de la durée du séjour en fonction du nombre total de séjours. Courbe de distribution normale de la durée du séjour.")

    col3, col4 = st.columns(2)

    dept_info = df_map[df_map["Département"] == selected_dept].iloc[0]
    dep_code = dept_info["dep_code"]
    
    df_sejour_filt = df_sejour[
        (df_sejour["dep_code"] == dep_code) &
        (df_sejour["Pathologie"] == pathologie_selected) &
        (df_sejour["ANNEE"] == year_selected)
    ]
    
    if not df_sejour_filt.empty:
        with col3:
            fig_sejour = px.bar(
                df_sejour_filt, 
                x="Durée séjour", 
                y="Nombre séjours",
                color="Durée séjour", 
                text="ratio durée du séjour",
                labels={"Nombre séjours": "Répartition (%)", "Durée séjour": "Durée des séjours (jours)"},
                title="Distribution durée du séjour"
            )

            fig_sejour.update_yaxes(
                range=[0, df_sejour_filt['Nombre séjours'].max() * 1.2]
                )
            
            fig_sejour.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside', width=1
                )
            
            fig_sejour.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_sejour, use_container_width=True)


        with col4:
            x = df_sejour_filt["Durée_num"]
            w = df_sejour_filt["Nombre séjours"]

            mu = np.average(x, weights=w)
            sigma = np.sqrt(np.average((x - mu) ** 2, weights=w))

            x_curve = np.linspace(min(x), max(x), 300)
            y_curve = norm.pdf(x_curve, mu, sigma)
            y_curve = y_curve * w.sum() / y_curve.sum()

            fig_gauss = go.Figure()

            fig_gauss.add_trace(go.Scatter(
                x=np.concatenate([x_curve, x_curve[::-1]]),
                y=np.concatenate([y_curve, np.zeros_like(y_curve)]),
               fill='toself',
              fillcolor='rgba(173,216,230,0.3)',
              line=dict(color='rgba(0,0,0,0)'),
                showlegend=False
            ))

            fig_gauss.add_trace(go.Scatter(
                x=x_curve, y=y_curve,
                mode="lines",
                name="Courbe normale",
                line=dict(color="blue", width=3)
            ))

            fig_gauss.add_vline(x=mu, line_dash="dash", line_color="red")
            fig_gauss.add_vline(x=mu + sigma, line_dash="dot", line_color="orange")

            fig_gauss.add_annotation(
               x=mu, y=max(y_curve) * 0.95,
               text=f"µ = {mu:.2f} j",
               showarrow=False,
              font=dict(color="red", size=14)
            )
            fig_gauss.add_annotation(
                x=mu + sigma, y=max(y_curve) * 0.85,
                text=f"µ + σ = {mu + sigma:.2f} j",
                showarrow=False,
                font=dict(color="orange", size=14)
        )

            fig_gauss.update_layout(
                xaxis_title="Durée du séjour (jours)",
                yaxis_title="Fréquence normalisée",
                title="Distribution normale de la durée de séjour",
                template="plotly_white",
                height=450,
                showlegend=False
            )
            st.plotly_chart(fig_gauss, use_container_width=True)
    else:
        st.info("Pas de données de durée de séjour disponibles.")


# =============================
# 5. Additional analyses
# =============================

st.markdown("---")
st.header("3️⃣ Analyses supplémentaires")

chart_option = st.selectbox(
    "Sélectionner une visualisation :",
    [
        "Répartition par sexe (France entière)",
        "Répartition par âge (France entière)",
        "Total par département",
    ],
    key="analysis_selector"
)

# -----------------------------
# SEX DISTRIBUTION (FRANCE)
# -----------------------------
if chart_option == "Répartition par sexe (France entière)":
    st.subheader(f"Répartition par sexe – {pathologie_selected} ({year_selected})")

    df_sex = df_tot_age[
        (df_tot_age["Pathologie"] == pathologie_selected) &
        (df_tot_age["ANNEE"] == year_selected)
    ].groupby("SEXE")["nbr recours"].sum().reset_index()
    
    df_sex["pct"] = (df_sex["nbr recours"] / df_sex["nbr recours"].sum() * 100).round(1)

    fig = px.bar(
        df_sex, x="SEXE", y="nbr recours", color="SEXE", text="pct",
        labels={"nbr recours": "Nombre de cas"},
        color_discrete_map={"Homme": "#318CE7", "Femme": "#DE3163"}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition="outside", width=0.3)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AGE DISTRIBUTION (FRANCE)
# -----------------------------
elif chart_option == "Répartition par âge (France entière)":
    st.subheader(f"Répartition par âge – {pathologie_selected} ({year_selected})")

    df_age = df_tranch_age[
        (df_tranch_age["Pathologie"] == pathologie_selected) &
        (df_tranch_age["ANNEE"] == year_selected)
    ].groupby("Tranche d'age")["nbr recours"].sum().reset_index()

    df_age["pct"] = (df_age["nbr recours"] / df_age["nbr recours"].sum() * 100).round(1)

    fig = px.bar(
        df_age, x="Tranche d'age", y="nbr recours",
        color="Tranche d'age", text="pct",
        labels={"nbr recours": "Nombre de cas"}
    )

    fig.update_traces(texttemplate='%{text:.1f}%', textposition="outside", width=0.7)
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TOTAL BY DEPARTMENT
# -----------------------------
elif chart_option == "Total par département":
    st.subheader(f"Nombre total de cas par département – {pathologie_selected} ({year_selected})")

    df_total_cases = df_tot_age[
        (df_tot_age["Pathologie"] == pathologie_selected) &
        (df_tot_age["ANNEE"] == year_selected)
    ].groupby("Département")["nbr recours"].sum().reset_index().sort_values("nbr recours", ascending=False)
    
    fig = px.bar(
        df_total_cases, x="Département", y="nbr recours",
        text="nbr recours",
        labels={"nbr recours": "Nombre de cas"}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
