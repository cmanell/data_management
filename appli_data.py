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

@st.cache_data
def load_data():
    df_tranch_age = pd.read_csv("df_tranch_age.csv", sep=",")
    df_tot_age    = pd.read_csv("df_tot_age.csv", sep=",")
    df_sejour     = pd.read_csv("df_sejour.csv", sep=",")
    df_tableau_1  = pd.read_csv("tableau_1.csv", sep=";")  # Add this line
    with open("departements.geojson", encoding="utf-8") as f:
        dep_geojson = json.load(f)
    return df_tranch_age, df_tot_age, df_sejour, df_tableau_1, dep_geojson

df_tranch_age, df_tot_age, df_sejour, df_tableau_1, dep_geojson = load_data()

# =============================
# 1. Page config
# =============================

st.set_page_config(page_title="Projet Data : Morbidités hospitalières", layout="wide")
st.title("Projet Data : Morbidité hospitalière en France métropolitaine + Corse")
st.markdown("### Taux de recours aux établissements de santé")
st.markdown("---")

# =============================
# 2. Sidebar filter
# =============================

st.sidebar.header("Filtres")
pathologies = sorted(df_tot_age['Pathologie'].drop_duplicates())
pathologie_selected = st.sidebar.selectbox("Pathologie :", pathologies, index=0)
years = sorted(df_tot_age['ANNEE'].drop_duplicates())
year_selected = st.sidebar.selectbox("Année :", years, index=len(years)-1)

# =============================
# 3. Main layout
# =============================

st.header("1️⃣ Vue d'ensemble par département")

col1, col2 = st.columns([1, 1])

# -----------------------------
# Left panel: Choropleth map
# -----------------------------

with col1:
    st.subheader("Carte interactive")
    
    df_map = df_tot_age[
        (df_tot_age["Pathologie"] == pathologie_selected) & 
        (df_tot_age["ANNEE"] == year_selected)
    ].copy()

    fig_map = px.choropleth(
        df_map,
        geojson=dep_geojson,
        locations="dep_code",
        featureidkey="properties.code",
        color="nbr recours",
        color_continuous_scale="Blues",
        range_color=(0, df_map["nbr recours"].max()),
        labels={"nbr recours": "Taux de recours"},
        hover_name="Département",
        hover_data={"Pathologie": True, "Département": False, "dep_code": False}
    )

    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        title=f"Taux de recours - {pathologie_selected}",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        height=500
    )

    # Display the map using standard st.plotly_chart
    st.plotly_chart(fig_map, use_container_width=True, key="choropleth_map")
    
    # Add a selectbox as fallback for department selection
    st.markdown("**Sélectionner un département :**")
    departments = sorted(df_map['Département'].unique())
    selected_dept = st.selectbox(
        "Choisir un département",
        options=["-- Aucun --"] + departments,
        key="dept_selector"
    )

# -----------------------------
# Right panel: Detailed charts
# -----------------------------

with col2:
    st.subheader("Détails par département")

    if selected_dept and selected_dept != "-- Aucun --":
        # Get department info
        dept_info = df_map[df_map["Département"] == selected_dept].iloc[0]
        dep_code = dept_info["dep_code"]
        dep_name = dept_info["Département"]
        
        st.markdown(f"**Département sélectionné :** {dep_name} (`{dep_code}`)")
        
        # Filter the data (ONLY ONCE)
        df_tot_age_filt = df_tot_age[
            (df_tot_age["dep_code"] == dep_code) &
            (df_tot_age["Pathologie"] == pathologie_selected) &
            (df_tot_age["ANNEE"] == year_selected)
        ]
        
        # Add total case count
        total_cases = df_tot_age_filt["nbr recours"].sum()
        st.metric("Nombre total de cas", f"{total_cases:,.0f}")

        # Sex distribution
        if not df_tot_age_filt.empty:
            df_sex = df_tot_age_filt.groupby("SEXE")["nbr recours"].sum().reset_index()
            df_sex['pct'] = (df_sex['nbr recours'] / df_sex['nbr recours'].sum() * 100).round(1)
            
            fig_sex = px.bar(
                df_sex, x='SEXE', y='nbr recours', color='SEXE', text='pct',
                color_discrete_map={"Homme": "#318CE7", "Femme": "#DE3163"},
                title="Répartition par sexe"
            )
            fig_sex.update_traces(texttemplate='%{text:.1f}%', textposition='outside', width=0.3)
            fig_sex.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_sex, use_container_width=True)

        # Age distribution
        df_tranch_filt = df_tranch_age[
            (df_tranch_age["dep_code"] == dep_code) &
            (df_tranch_age["Pathologie"] == pathologie_selected) &
            (df_tranch_age["ANNEE"] == year_selected)
        ]
        
        if not df_tranch_filt.empty:
            df_age = df_tranch_filt.groupby("Tranche d'age")["nbr recours"].sum().reset_index()
            df_age['pct'] = (df_age['nbr recours'] / df_age['nbr recours'].sum() * 100).round(1)
            
            fig_age = px.bar(
                df_age, x="Tranche d'age", y='nbr recours',
                color="Tranche d'age", text='pct',
                title="Répartition par tranche d'âge"
            )
            fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_age.update_yaxes(range=[0, df_age['nbr recours'].max() * 1.15])
            fig_age.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Pas de données détaillées pour ce département.")
    else:
        st.info("👈 Sélectionnez un département dans la liste pour voir les détails.")


# =============================
# 4. Duration of stay section
# =============================

if selected_dept and selected_dept != "-- Aucun --":
    st.markdown("---")
    st.header("2️⃣ Analyse de la durée de séjour")
    
    dept_info = df_map[df_map["Département"] == selected_dept].iloc[0]
    dep_code = dept_info["dep_code"]
    
    df_sejour_filt = df_sejour[
        (df_sejour["dep_code"] == dep_code) &
        (df_sejour["Pathologie"] == pathologie_selected) &
        (df_sejour["ANNEE"] == year_selected)
    ]
    
    if not df_sejour_filt.empty:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Distribution des durées")
            fig_sejour = px.bar(
                df_sejour_filt, x="Durée séjour", y="Nombre séjours",
                color="Durée séjour", text="ratio durée du séjour",
                labels={"Nombre séjours": "Nombre de séjours", "Durée séjour": "Durée (jours)"}
            )
            fig_sejour.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig_sejour, use_container_width=True)

        with col_b:
            st.subheader("Distribution normale théorique")
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
                template="plotly_white"
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
        "Répartition par sexe (France)",
        "Répartition par âge (France)",
        "Total par département",
        "Analyse de risque (Sexe vs Pathologie)",
        "Analyse de risque (Âge vs Pathologie)",
        "Analyse de risque (Département vs Pathologie)"
    ],
    key="analysis_selector"
)

# -----------------------------
# SEX DISTRIBUTION (FRANCE)
# -----------------------------
if chart_option == "Répartition par sexe (France)":
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
    fig.update_traces(texttemplate='%{text:.1f}%', textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AGE DISTRIBUTION (FRANCE)
# -----------------------------
elif chart_option == "Répartition par âge (France)":
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
    fig.update_traces(texttemplate='%{text:.1f}%', textposition="outside")
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
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)



    # -----------------------------
# RISK ANALYSIS: SEX vs PATHOLOGY
# -----------------------------
elif chart_option == "Analyse de risque (Sexe vs Pathologie)":
    st.subheader(f"Analyse de risque : Sexe vs Pathologie ({year_selected})")
    
    # Calculate total cases by sex and pathology
    df_risk = df_tot_age[df_tot_age["ANNEE"] == year_selected].groupby(
        ["SEXE", "Pathologie"]
    )["nbr recours"].sum().reset_index()
    
    # Calculate total for each sex
    df_sex_total = df_risk.groupby("SEXE")["nbr recours"].sum().reset_index()
    df_sex_total.columns = ["SEXE", "total_sexe"]
    
    # Merge to get percentages
    df_risk = df_risk.merge(df_sex_total, on="SEXE")
    df_risk["percentage"] = (df_risk["nbr recours"] / df_risk["total_sexe"] * 100).round(2)
    
    # Pivot for comparison
    df_pivot = df_risk.pivot(index="Pathologie", columns="SEXE", values="percentage").reset_index()
    
    if "Homme" in df_pivot.columns and "Femme" in df_pivot.columns:
        df_pivot["Différence (H-F)"] = df_pivot["Homme"] - df_pivot["Femme"]
        df_pivot = df_pivot.sort_values("Différence (H-F)", ascending=False)
        
        # Highlight selected pathology
        df_pivot["Couleur"] = df_pivot["Pathologie"].apply(
            lambda x: "Sélectionné" if x == pathologie_selected else "Autre"
        )
        
        fig = px.bar(
            df_pivot, x="Pathologie", y="Différence (H-F)",
            color="Couleur",
            color_discrete_map={"Sélectionné": "#FF6B6B", "Autre": "#4ECDC4"},
            labels={"Différence (H-F)": "Différence de risque (% Hommes - % Femmes)"},
            title="Différence de prévalence entre hommes et femmes par pathologie"
        )
        fig.update_xaxes(tickangle=45)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interprétation :**
        - Valeurs positives : pathologie plus fréquente chez les hommes
        - Valeurs négatives : pathologie plus fréquente chez les femmes
        - La pathologie sélectionnée est mise en évidence en rouge
        """)
        
        # Show detailed table for selected pathology
        selected_row = df_pivot[df_pivot["Pathologie"] == pathologie_selected]
        if not selected_row.empty:
            st.markdown(f"**Détails pour {pathologie_selected} :**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("% Hommes", f"{selected_row['Homme'].values[0]:.2f}%")
            with col2:
                st.metric("% Femmes", f"{selected_row['Femme'].values[0]:.2f}%")
            with col3:
                diff = selected_row['Différence (H-F)'].values[0]
                st.metric("Différence", f"{diff:.2f}%", 
                         delta=None if abs(diff) < 1 else ("Plus fréquent chez hommes" if diff > 0 else "Plus fréquent chez femmes"))

# -----------------------------
# RISK ANALYSIS: AGE vs PATHOLOGY
# -----------------------------
elif chart_option == "Analyse de risque (Âge vs Pathologie)":
    st.subheader(f"Analyse de risque : Âge vs Pathologie ({year_selected})")
    
    # Calculate cases by age and pathology
    df_risk = df_tranch_age[df_tranch_age["ANNEE"] == year_selected].groupby(
        ["Tranche d'age", "Pathologie"]
    )["nbr recours"].sum().reset_index()
    
    # Calculate total for each age group
    df_age_total = df_risk.groupby("Tranche d'age")["nbr recours"].sum().reset_index()
    df_age_total.columns = ["Tranche d'age", "total_age"]
    
    # Merge and calculate percentages
    df_risk = df_risk.merge(df_age_total, on="Tranche d'age")
    df_risk["percentage"] = (df_risk["nbr recours"] / df_risk["total_age"] * 100).round(2)
    
    # Filter for selected pathology
    df_selected = df_risk[df_risk["Pathologie"] == pathologie_selected]
    
    fig = px.bar(
        df_selected, x="Tranche d'age", y="percentage",
        color="Tranche d'age",
        text="percentage",
        labels={"percentage": "% des cas dans la tranche d'âge"},
        title=f"Distribution du risque par tranche d'âge - {pathologie_selected}"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap for all pathologies
    st.markdown("**Comparaison entre toutes les pathologies :**")
    df_heatmap = df_risk.pivot(index="Pathologie", columns="Tranche d'age", values="percentage")
    
    fig_heat = px.imshow(
        df_heatmap,
        labels=dict(x="Tranche d'âge", y="Pathologie", color="% de prévalence"),
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_heat.update_xaxes(side="bottom")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("""
    **Interprétation :**
    - Les couleurs plus foncées indiquent une prévalence plus élevée
    - Permet d'identifier les pathologies spécifiques à certaines tranches d'âge
    """)

# -----------------------------
# RISK ANALYSIS: DEPARTMENT vs PATHOLOGY
# -----------------------------
elif chart_option == "Analyse de risque (Département vs Pathologie)":
    st.subheader(f"Analyse de risque : Département vs Pathologie ({year_selected})")
    
    # Calculate cases by department and pathology
    df_risk = df_tot_age[df_tot_age["ANNEE"] == year_selected].groupby(
        ["Département", "Pathologie"]
    )["nbr recours"].sum().reset_index()
    
    # Calculate total for each department
    df_dept_total = df_risk.groupby("Département")["nbr recours"].sum().reset_index()
    df_dept_total.columns = ["Département", "total_dept"]
    
    # Merge and calculate percentages
    df_risk = df_risk.merge(df_dept_total, on="Département")
    df_risk["percentage"] = (df_risk["nbr recours"] / df_risk["total_dept"] * 100).round(2)
    
    # Calculate national average for selected pathology
    df_selected = df_risk[df_risk["Pathologie"] == pathologie_selected].copy()
    national_avg = df_selected["percentage"].mean()
    
    df_selected["Écart à la moyenne"] = df_selected["percentage"] - national_avg
    df_selected = df_selected.sort_values("Écart à la moyenne", ascending=False)
    
    # Highlight if a department is selected
    if selected_dept and selected_dept != "-- Aucun --":
        df_selected["Couleur"] = df_selected["Département"].apply(
            lambda x: "Sélectionné" if x == selected_dept else "Autre"
        )
        color_map = {"Sélectionné": "#FF6B6B", "Autre": "#4ECDC4"}
    else:
        df_selected["Couleur"] = "Standard"
        color_map = {"Standard": "#4ECDC4"}
    
    fig = px.bar(
        df_selected, x="Département", y="Écart à la moyenne",
        color="Couleur",
        color_discrete_map=color_map,
        labels={"Écart à la moyenne": "Écart à la moyenne nationale (%)"},
        title=f"Écart de prévalence par département - {pathologie_selected}"
    )
    fig.update_xaxes(tickangle=45)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text=f"Moyenne nationale: {national_avg:.2f}%")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **Interprétation :**
    - Moyenne nationale : **{national_avg:.2f}%**
    - Valeurs positives : départements avec prévalence supérieure à la moyenne
    - Valeurs négatives : départements avec prévalence inférieure à la moyenne
    """)
    
    # Show top and bottom departments
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 départements (prévalence la plus élevée) :**")
        top5 = df_selected.nlargest(5, "percentage")[["Département", "percentage", "Écart à la moyenne"]]
        st.dataframe(top5, hide_index=True)
    
    with col2:
        st.markdown("**Top 5 départements (prévalence la plus faible) :**")
        bottom5 = df_selected.nsmallest(5, "percentage")[["Département", "percentage", "Écart à la moyenne"]]
        st.dataframe(bottom5, hide_index=True)