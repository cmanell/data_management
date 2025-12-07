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
        "Test du Chi² (Sexe vs Pathologie)",
        "Test du Chi² (Âge vs Pathologie)",
        "Test du Chi² (Département vs Pathologie)"
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
# CHI-SQUARED TESTS
# -----------------------------
elif chart_option.startswith("Test du Chi²"):
    st.subheader(f"{chart_option} ({year_selected})")
    
    # Explanation box
    with st.expander("ℹ️ Qu'est-ce que le test du Chi² ?"):
        st.markdown("""
        ### Le Test du Chi² (Chi-carré) d'indépendance
        
        **Objectif :** Déterminer s'il existe une relation statistiquement significative entre deux variables catégorielles.
        
        **Comment ça marche ?**
        1. On compare les fréquences **observées** (les données réelles) aux fréquences **attendues** (si les variables étaient indépendantes)
        2. Le Chi² mesure l'écart entre ces deux distributions
        3. La **p-value** indique la probabilité que cet écart soit dû au hasard
        
        **Interprétation :**
        - **p < 0.05** : Les variables sont **liées** (relation significative) ✔️
        - **p ≥ 0.05** : Les variables sont **indépendantes** (pas de relation) ❌
        
        **Exemple :** 
        - Si le test Sexe vs Pathologie est significatif, cela signifie que certaines pathologies affectent différemment les hommes et les femmes.
        - Si p = 0.001, il y a seulement 0.1% de chance que cette différence soit due au hasard.
        """)
    
    st.markdown("---")
    
    # -----------------------------
    # SEX VS PATHOLOGY
    # -----------------------------
    if chart_option == "Test du Chi² (Sexe vs Pathologie)":
        st.subheader("Sexe vs Pathologie")
        st.info("Ce test détermine si certaines pathologies affectent différemment les hommes et les femmes.")
        
        # Filter data
        df_filtered = df_tableau_1[
            (df_tableau_1["ANNEE"] == year_selected) &
            (~df_tableau_1["SEXE"].str.contains("Ensemble", na=False))
        ].copy()
        
        # Convert to numeric
        df_filtered['ind_freq'] = pd.to_numeric(df_filtered['ind_freq'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['ind_freq'])
        
        # Clean pathology names (remove codes)
        df_filtered['PATHOLOGIE_CLEAN'] = df_filtered['PATHOLOGIE'].str.replace(r'^\d+-', '', regex=True)
        
        if not df_filtered.empty:
            # Create contingency table
            cont_table = df_filtered.pivot_table(
                index="SEXE",
                columns="PATHOLOGIE_CLEAN",
                values="ind_freq",
                aggfunc="sum"
            ).fillna(0)
            
            if cont_table.shape[0] > 1 and cont_table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(cont_table)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Chi²", f"{chi2:.2f}")
                col2.metric("p-value", f"{p:.2e}")
                col3.metric("Degrés de liberté", dof)
                
                if p < 0.05:
                    st.success("✔️ **Résultat significatif** : Le sexe et la pathologie sont statistiquement liés (p < 0.05)")
                    st.markdown("**Interprétation :** Certaines pathologies affectent significativement plus un sexe que l'autre.")
                else:
                    st.info("ℹ️ **Pas de lien significatif** : Le sexe et la pathologie semblent indépendants (p ≥ 0.05)")
                
                if st.checkbox("Afficher la table de contingence", key="chi_sex_patho"):
                    st.dataframe(cont_table.style.format("{:.0f}"))
            else:
                st.warning("⚠️ Données insuffisantes pour effectuer le test du Chi².")
        else:
            st.warning("⚠️ Aucune donnée disponible.")
    
    # -----------------------------
    # AGE VS PATHOLOGY
    # -----------------------------
    elif chart_option == "Test du Chi² (Âge vs Pathologie)":
        st.subheader("Âge vs Pathologie")
        st.info("Ce test détermine si certaines pathologies sont plus fréquentes dans certaines tranches d'âge.")
        
        # Filter data
        df_filtered = df_tableau_1[
            (df_tableau_1["ANNEE"] == year_selected) &
            (~df_tableau_1["Tranche d'age"].str.contains("Ensemble|Tous", na=False, case=False))
        ].copy()
        
        # Convert to numeric
        df_filtered['ind_freq'] = pd.to_numeric(df_filtered['ind_freq'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['ind_freq'])
        
        # Clean pathology names (remove codes)
        df_filtered['PATHOLOGIE_CLEAN'] = df_filtered['PATHOLOGIE'].str.replace(r'^\d+-', '', regex=True)
        
        if not df_filtered.empty:
            # Create contingency table
            cont_table = df_filtered.pivot_table(
                index="Tranche d'age",
                columns="PATHOLOGIE_CLEAN",
                values="ind_freq",
                aggfunc="sum"
            ).fillna(0)
            
            if cont_table.shape[0] > 1 and cont_table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(cont_table)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Chi²", f"{chi2:.2f}")
                col2.metric("p-value", f"{p:.2e}")
                col3.metric("Degrés de liberté", dof)
                
                if p < 0.05:
                    st.success("✔️ **Résultat significatif** : L'âge et la pathologie sont statistiquement liés (p < 0.05)")
                    st.markdown("**Interprétation :** Certaines pathologies sont significativement plus fréquentes dans certaines tranches d'âge.")
                else:
                    st.info("ℹ️ **Pas de lien significatif** : L'âge et la pathologie semblent indépendants (p ≥ 0.05)")
                
                if st.checkbox("Afficher la table de contingence", key="chi_age_patho"):
                    st.dataframe(cont_table.style.format("{:.0f}"))
            else:
                st.warning("⚠️ Données insuffisantes pour effectuer le test du Chi².")
        else:
            st.warning("⚠️ Aucune donnée disponible.")
    
    # -----------------------------
    # DEPARTMENT VS PATHOLOGY
    # -----------------------------
    elif chart_option == "Test du Chi² (Département vs Pathologie)":
        st.subheader("Département vs Pathologie")
        st.info("Ce test détermine si certaines pathologies ont une distribution géographique particulière.")
        
        # Filter data
        df_filtered = df_tableau_1[
            (df_tableau_1["ANNEE"] == year_selected)
        ].copy()
        
        # Convert to numeric
        df_filtered['ind_freq'] = pd.to_numeric(df_filtered['ind_freq'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['ind_freq'])
        
        # Clean pathology names (remove codes)
        df_filtered['PATHOLOGIE_CLEAN'] = df_filtered['PATHOLOGIE'].str.replace(r'^\d+-', '', regex=True)
        
        if not df_filtered.empty:
            # Create contingency table
            cont_table = df_filtered.pivot_table(
                index="ZONE",
                columns="PATHOLOGIE_CLEAN",
                values="ind_freq",
                aggfunc="sum"
            ).fillna(0)
            
            if cont_table.shape[0] > 1 and cont_table.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(cont_table)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Chi²", f"{chi2:.2f}")
                col2.metric("p-value", f"{p:.2e}")
                col3.metric("Degrés de liberté", dof)
                
                if p < 0.05:
                    st.success("✔️ **Résultat significatif** : Le département et la pathologie sont statistiquement liés (p < 0.05)")
                    st.markdown("**Interprétation :** Certaines pathologies ont une distribution géographique particulière (facteurs environnementaux, démographiques, etc.).")
                else:
                    st.info("ℹ️ **Pas de lien significatif** : Le département et la pathologie semblent indépendants (p ≥ 0.05)")
                
                if st.checkbox("Afficher la table de contingence", key="chi_dept_patho"):
                    st.dataframe(cont_table.style.format("{:.0f}"))
            else:
                st.warning("⚠️ Données insuffisantes pour effectuer le test du Chi².")
        else:
            st.warning("⚠️ Aucune donnée disponible.")
                