import plotly.express as px # pour ploter des figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd         # permettant la création de fataframe
import json                 # pour lire les jason
from streamlit_plotly_events import plotly_events

df_tranch_age = pd.read_csv('df_tranch_age.csv', sep = ',')
df_tot_age = pd.read_csv('df_tot_age.csv', sep = ',')


### Représentations graphiques ###
with open("departements.geojson", encoding="utf-8") as f:
    dep_geojson = json.load(f)


st.set_page_config(page_title="Projet Data : Morbidités hospitalières", layout="wide")
st.title("Projet Data : Morbidité hospitalière")
st.markdown("## Taux de recours aux établissements de santé en France")
st.markdown("## 1 - Analyse par départements et évolution dans le temps")
st.markdown("---") 

selected_points = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition cartographique pathologique")
    pathologies_tout = sorted(df_tot_age["Pathologie"].drop_duplicates())

    pathologie_selected = st.selectbox(
        "Pathologie sélectionnée :",
        pathologies_tout
    )

    if pathologie_selected:

        df_p = df_tot_age[(df_tot_age["Pathologie"] == pathologie_selected)].copy()
        fig = px.choropleth(
            df_p,
            geojson = dep_geojson,
            locations="dep_code",              
            featureidkey="properties.code",    
            color="nbr recours",
            animation_frame="ANNEE",           
            color_continuous_scale="Blues",
            range_color=(0, df_p["nbr recours"].max()),
            labels={"nbr recours": "nbr recours"},
            hover_name="Département",
            hover_data={"Pathologie": True, "Département": False}
        )

        fig.update_geos(
            fitbounds="locations",
            visible=False
        )

        fig.update_layout(
            title=f"Nombre de cas de {pathologie_selected} par département (H+F)",
            margin={"r":0,"t":40,"l":0,"b":0})
        
        # selectionner le département et conserver le code_dep
        selected_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=500,
            override_width="100%"
        )
    else:
        st.info("Select a pathologie.")


with col2:
    st.subheader("Répartition des sexes selon la pathologie")

    if selected_points:
    
        # Pour un choropleth, le code du département est dans "location"
        idx = selected_points[0]["pointNumber"]  
        dep_code = df_p.iloc[idx]["dep_code"]
        st.markdown(f"#### Département sélectionné : `{dep_code}`")


        annees = sorted(df_tot_age["ANNEE"].drop_duplicates())
        annee_selected = st.selectbox("Année :", annees, index=len(annees)-1)

        st.markdown(f"Nombre de recours aux structures de santé pour la pathologie {pathologie_selected}"
                f"dans le département {dep_code} pour l'année {annee_selected} par tranche d'age.") 

        if annee_selected:
           df_tot_age_filt = df_tot_age[
               (df_tot_age["dep_code"] == dep_code) &
               (df_tot_age["Pathologie"] == pathologie_selected) &
               (df_tot_age["ANNEE"] == annee_selected)
           ].copy()

           if not df_tot_age_filt.empty:
               fig1 = px.bar(
                   df_tot_age_filt,
                   y='nbr recours',
                   x='SEXE',
                   color='SEXE',
                   title=("Graphique représentant la distribution selon le sexe"),
                   labels={
                       'nbr recours': 'nombre de cas',
                       'SEXE': 'Sexe'
                   },
                   barmode='group',
                   text='ratio par sexe',  
                   color_discrete_map={
                       "Homme": "#318CE7",
                       "Femme": "#DE3163"
                   }
               )
               y_range1 = df_tot_age_filt['nbr recours'].max()*(1.1)
               fig1.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
               fig1.update_traces(width=0.3)  
               fig1.update_layout(
               bargap=1,       
               )
               fig1.update_layout(legend_title_text="Légende")
               fig1.update_yaxes(range=[0, y_range1])

               st.plotly_chart(fig1, use_container_width=True)
           else:
               st.info("Pas de données détaillées pour ce département.")
    else:
        st.info("Cliquez sur un département de la carte pour afficher les graphes")

    st.markdown("---")
    st.subheader("Répartition des tranches d'âge selon la pathologie")

    if selected_points:
        if annee_selected:
            df_tranch_filt = df_tranch_age[
               (df_tranch_age["dep_code"] == dep_code) &
               (df_tranch_age["Pathologie"] == pathologie_selected) &
               (df_tranch_age["ANNEE"] == annee_selected)
            ].copy()

            if not df_tranch_filt.empty:

                fig2 = px.bar(
                    df_tranch_filt,
                    y='nbr recours', 
                    x= 'Tranche d\'age',
                    color='Tranche d\'age', 
                    title=("Graphique de répartition selon la tranche d'âge"),            
                    labels={
                        'nbr recours': 'nombre de cas', 
                        'Tranche d\'âge': 'Tranche d\'âge'}, 
                    barmode='group', 
                    text='ratio par tranche d\'age',      
                    )
                y_range2 = df_tranch_filt['nbr recours'].max()*(1.1)
                fig2.update_layout(legend_title_text="Légende")
                fig2.update_traces(texttemplate='%{text:.2s}%', textposition='outside')
                fig2.update_traces(width=0.6)  
                fig2.update_yaxes(range=[0, y_range2])

                st.plotly_chart(fig2, use_container_width=True)
            else:
               st.info("Pas de données détaillées pour ce département.")
        else:
            st.info("Cliquez sur un département de la carte pour afficher les graphes")

