# 🏥 Analyse des morbidités hospitalières en France  
### 📊 Projet collaboratif – Data Management & Application interactive (Python – Streamlit)

### 👥 Collaborateurs
- **Cédric MANELLI**
- _(à compléter)_
- _(à compléter)_

---

## 📌 Description du projet

Ce projet a pour objectif d’analyser les **taux de recours aux établissements de santé en France** à partir des données de **morbidité hospitalière (MCO)**.  
L’étude vise à explorer les hospitalisations en fonction :

- des **pathologies**
- du **sexe**
- des **tranches d’âge**
- de l’**année**
- du **département**
- de la **durée des séjours**

Les résultats sont intégrés dans une **application interactive Streamlit** permettant d’explorer les données de manière visuelle, dynamique et intuitive (filtres, clic sur la carte, animations temporelles, etc.).
---

## 🗂️ Source des données

📎 **Morbidité hospitalière (MCO)** – Produite par l’**ATIH** et publiée par la **DREES**  
🔗 https://www.data.gouv.fr/api/1/datasets/r/adba3d85-ad73-41d9-b152-e0f3f8153db5

📆 **Période étudiée : 2018 → 2022**  
🌍 **Échelle : national → régional → départemental (France métropolitaine)**  
📄 **Formats disponibles : CSV, JSON, Parquet**

### 📍 Description officielle

Les données portent sur les hospitalisations en **soins de courte durée (MCO)**.  
La dataviz officielle propose 3 tableaux :

1. **Répartition des séjours** selon le sexe, l’âge et la pathologie traitée  
2. **Durée des séjours et durée moyenne** selon la pathologie  
3. **Taux de recours** selon le sexe, l’âge et la pathologie

---

## 🎯 Objectifs du projet

### 🔎 1) Exploration & Préparation
- Compréhension de la structure de la base
- Nettoyage :  
  - gestion des valeurs manquantes (moyenne locale)
  - conversions de types
  - harmonisation des pathologies
  - extraction de `dep_code` (incluant Corse **2A/2B**)
- Sélection des dimensions pertinentes : sexe, âge, pathologie, durée, géographie, temporalité…

### 🧮 2) Agrégations & Calculs avancés
Création de plusieurs jeux de données :

| Dataset | Contenu |
|---------|--------|
| `df_tot_age.csv` | Agrégations par sexe, pathologie, année, département |
| `df_tranch_age.csv` | Agrégations par tranche d’âge |
| `df_sejour.csv` | Analyse des durées de séjour |

🔢 Calculs statistiques intégrés :
- **Ratio Homme / Femme**
- **Pourcentage par tranche d’âge**
- Conversion en durée numérique (`Durée_num`)
- **Moyennes et écarts-types pondérés**

#### 📌 Formules utilisées

Moyenne pondérée :
\[
\mu = \frac{\sum w_i x_i}{\sum w_i}
\]

Écart-type pondéré :
\[
\sigma = \sqrt{\frac{\sum w_i (x_i - \mu)^2}{\sum w_i}}
\]

💡 Ces résultats sont utilisés pour tracer une **distribution normale théorique** superposée aux données réelles.

---

## 🖥 Application Streamlit interactive

### 🗺 Carte choroplèthe dynamique
- Visualisation par **département**
- Filtre par **pathologie**
- Animation **par année**
- Clic sur un département → **affichage des analyses dédiées**

### 📊 Analyses proposées

| Analyse | Description |
|---------|-------------|
| 📉 Durée des séjours | Histogrammes + **courbe de Gauss (µ & σ)** |
| 🚻 Répartition par sexe | Barres comparatives + **ratio affiché** |
| 👶👵 Répartition par tranche d’âge | Histogramme + **pourcentages** |
| 🗺 Évolution géographique | Carte animée (département) |

---

## ⚙️ Installation

### 🧰 Prérequis
- Python **3.10+**
- `pip` ou `conda`

### 📦 Installation des dépendances
```bash
pip install -r requirements.txt
