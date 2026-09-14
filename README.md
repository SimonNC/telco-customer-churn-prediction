# Prédiction de churn client Telco - Aide à la décision ML

[🇬🇧 English version](README.en.md)

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://telco-customer-churn-prediction-simonnc.streamlit.app/)

> **0.84 ROC-AUC, 0.73 de recall sur les churners.** EDA orientée métier sur 7 043 clients, pipeline ML sans fuite de données, et application Streamlit déployée pour la simulation de rétention.

---

## 📌 Contexte métier

Le churn client est un enjeu critique pour les entreprises de télécommunications : acquérir un nouveau client coûte significativement plus cher que de fidéliser un client existant. La capacité à **identifier de manière proactive les clients à risque** et à soutenir des **stratégies de rétention data-driven** constitue un levier de valeur business direct.

Ce projet démontre comment un Data Analyst peut faire le lien entre **analyse exploratoire des données**, **modélisation statistique** et **recommandations métier actionnables** pour traiter ce problème de bout en bout.

---

## 🔗 Démo en ligne

👉 **https://telco-customer-churn-prediction-simonnc.streamlit.app/**

L'application Streamlit déployée permet aux utilisateurs de :

- Simuler des profils clients individuels
- Estimer la probabilité de churn à l'aide du modèle Random Forest entraîné
- Ajuster le seuil de décision selon la stratégie métier (coût d'acquisition vs. rétention)
- Recevoir des recommandations de rétention actionnables

---

## 🎯 Objectif du projet

Construire un modèle de classification binaire interprétable et prêt pour la production, capable de prédire le churn client, avec une **stratégie recall-first** pour minimiser les clients à haut risque manqués.

L'approche est délibérément **orientée métier** : les résultats sont conçus pour soutenir des décisions opérationnelles, et non pour démontrer des techniques ML académiques.

---

## 📊 Jeu de données

| Attribut | Valeur |
|---|---|
| **Source** | Telco Customer Churn (IBM / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| **Granularité** | 1 ligne = 1 client |
| **Taille** | 7 043 clients |
| **Cible** | `Churn` (Yes / No) |
| **Taux de churn** | ~26,5% |

---

## 🧠 Méthodologie

### 1. Analyse exploratoire des données (EDA)

Analyse orientée métier pour identifier les principaux facteurs de churn avant toute modélisation. L'objectif est de comprendre **pourquoi les clients partent**, pas seulement de prédire **qui** partira.

### 2. Feature Engineering

Sélection de variables guidée par le métier, encodage adapté, et **pipelines de prétraitement sans fuite de données** pour garantir la fiabilité du modèle en production.

### 3. Modélisation

| Modèle | ROC-AUC | Recall (Churn) | Precision (Churn) |
|---|---|---|---|
| Logistic Regression (référence) | ~0.80 | ~0.67 | ~0.52 |
| **Random Forest (final)** | **0.84** | **0.73** | 0.56 |

**Choix du modèle** : Random Forest a été retenu pour son recall et son pouvoir discriminant supérieurs, alignés avec les objectifs de prévention du churn où manquer un client à haut risque coûte plus cher qu'une fausse alerte.

### 4. Déploiement

Pipeline final (prétraitement + modèle) exporté et intégré dans une application Streamlit garantissant la cohérence des prédictions avec l'évaluation hors ligne.

---

## 📈 Enseignements clés

| Constat | Implication métier |
|---|---|
| Les **contrats mensuels** présentent un churn significativement plus élevé | L'engagement contractuel réduit le risque de churn |
| Le risque de churn culmine durant les **premiers mois** du cycle de vie client | L'onboarding et l'engagement précoce sont critiques |
| L'absence de **Tech Support** est fortement associée à un churn plus élevé | Le bundling de services comme levier de rétention |
| L'**Online Security** agit comme un puissant levier de rétention chez les clients internet | Opportunité d'upsell ciblé |

Ces enseignements sont **actionnables sans le modèle** : ils orientent les stratégies produit, tarification et succès client indépendamment des prédictions ML.

---

## 🗂️ Structure du projet

```
telco-customer-churn-prediction/
├── data/
│   ├── raw/                       # Jeu de données brut
│   └── processed/                 # Données nettoyées
├── notebooks/
│   ├── 01_eda.ipynb               # Analyse exploratoire
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb          # Entraînement & évaluation du modèle
├── src/
│   └── data_prep.py               # Prétraitement réutilisable
├── models/
│   └── churn_model_rf.joblib      # Pipeline du modèle entraîné
├── app/
│   └── streamlit_app.py           # Application d'aide à la décision
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Exécution en local

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install --upgrade pip
pip install -r requirements.txt

# Lancer les notebooks
jupyter notebook

# Lancer l'application Streamlit
streamlit run app/streamlit_app.py
```

---

## 🔍 Limites et pistes d'amélioration

- Ajustement du seuil en fonction de la valeur vie client et du coût de rétention
- Analyse de l'importance des variables pour une interprétabilité métier plus poussée
- Optimisation cost-sensitive ou basée sur le profit
- Suivi de la performance du modèle dans le temps (détection de data drift)

---

## 🎯 Compétences démontrées

Ce projet démontre des compétences alignées avec les exigences du marché pour un poste de **Data Analyst** :

| Compétence | Comment elle est démontrée |
|---|---|
| **Analyse exploratoire des données (EDA)** | Analyse orientée métier de 7 043 clients pour identifier les facteurs de churn |
| **Modélisation statistique** | Référence Logistic Regression + Random Forest avec stratégie recall-first |
| **Python** (Pandas, Scikit-learn) | Pipeline de bout en bout, des données brutes au modèle déployé |
| **Visualisation de données** | Graphiques et résultats conçus pour communiquer avec les parties prenantes métier |
| **Analyse des besoins métier** | Problème formulé du point de vue opérationnel (coût de rétention vs. coût d'acquisition) |
| **Conception de KPI** | Taux de churn, leviers de rétention et ajustement de seuil comme métriques d'aide à la décision |
| **Déploiement** | Application Streamlit pour la simulation client en temps réel et le test de stratégies de rétention |

---

## 🔗 Projets liés

Ce projet complète le portfolio aux côtés de :

- 👉 [Olist E-commerce: End-to-End BI Solution](https://github.com/SimonNC/olist-data-analysis) (Python + tableaux de bord Power BI)
- 👉 [Olist Analytics Engineering Pipeline](https://github.com/SimonNC/olist-dbt-duckdb) (SQL + dbt)

---

## 👤 Auteur

**Simon Jorite**
Data Analyst - [Certifié Microsoft Power BI Data Analyst (PL-300)](https://learn.microsoft.com/en-us/users/simonjorite-4846/credentials/b2cc3310a92a9302)

15 ans d'expérience en finance, opérations et e-commerce. Je transforme des jeux de données complexes en KPI fiables et en tableaux de bord prêts pour la décision.

- GitHub : [github.com/SimonNC](https://github.com/SimonNC)
- LinkedIn : [linkedin.com/in/simonjorite](https://www.linkedin.com/in/simonjorite)
- Email : simon.jorite@gmail.com
- Localisation : Lyon, France (Ouvert à un poste hybride ou en télétravail)
- Prise de RDV : [Réserver un échange de 30 min](https://calendly.com/simon-jorite/echange-da)
