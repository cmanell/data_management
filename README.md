# Projet collaboratif - Data management Python

Collaborateurs:

- Sufyan Nadat
- Jacques Allison
- Cédric MANELLI

## Description

Ce projet a pour objectif d’analyser les **taux de recours aux établissements de santé en France** à partir des données de morbidité hospitalière, puis de proposer une **application interactive Streamlit** permettant d’explorer les résultats par pathologie, département, année, sexe et tranche d’âge.   

---

## Source des données

**Base utilisée : Morbidité hospitalière (MCO)**  
URL : https://www.data.gouv.fr/api/1/datasets/r/adba3d85-ad73-41d9-b152-e0f3f8153db5

### Description officielle

Il s'agit des données sur les hospitalisations en court séjour survenues entre 2018 et 2022.  
Ces tableaux ont été réalisés à partir des données hospitalières (PMSI MCO) produites par l’Agence technique de l’information sur l’hospitalisation (ATIH).  

Ils complètent la série de données annuelles mises à disposition sur le site data.Drees depuis 2010, pouvant être déclinés au niveau :

- national  
- régional  
- départemental  

La dataviz d’origine propose 3 tableaux :

1. Répartition des séjours dans les établissements de soins de courte durée (MCO) selon le sexe, l’âge des patients et la pathologie traitée  
2. Répartition des séjours dans les établissements de soins de courte durée (MCO) selon la durée du séjour et la pathologie traitée, et durée moyenne de séjour  
3. Taux de recours aux établissements de soins de courte durée (MCO) selon le sexe, l’âge des patients et la pathologie traitée  

Les données sont disponibles en CSV, JSON et Parquet.  

---

## Objectifs du projet

1. **Exploration et préparation des données**  
   - Comprendre la structure de la base (dimensions, dictionnaire des variables)  
   - Identifier les axes d’analyse pertinents : pathologie, âge, sexe, département, année…  
   - Nettoyer les données (valeurs manquantes, types, filtrage)  

2. **Construction de tables agrégées**  
   - Création de jeux de données intermédiaires pour l’analyse :  
     - `df_tot_age.csv` : agrégations par département, pathologie, sexe, année, etc.  
     - `df_tranch_age.csv` : agrégations par tranches d’âge, département, pathologie, année, etc.  

3. **Mise en place d’une application Streamlit interactive**  
   - Visualisation cartographique des taux de recours par **département**  
   - Exploration détaillée par **sexe** et **tranches d’âge** pour une pathologie donnée  
   - Interaction via clic sur la carte + filtres dans l’interface. :contentReference[oaicite:1]{index=1}  

---

## Structure du projet

```text
.
├── appli_data.py        # Application Streamlit (visualisation interactive)
├── traitement.ipynb     # Notebook de préparation et d’exploration des données
├── df_tot_age.csv       # Données agrégées par pathologie / sexe / départements / année
├── df_tranch_age.csv    # Données agrégées par tranches d’âge / départements / année
├── departements.geojson # Polygones des départements français (pour la carte choroplèthe)
└── README.md            # Présentation du projet (ce fichier)
