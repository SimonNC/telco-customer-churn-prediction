# Telco Customer Churn Prediction - ML Decision Support

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://telco-customer-churn-prediction-simonnc.streamlit.app/)

> **0.84 ROC-AUC, 0.73 recall on churners.** Business-oriented EDA on 7,043 customers, leakage-safe ML pipeline, and a deployed Streamlit app for retention simulation.

---

## 📌 Business Context

Customer churn is a critical challenge for telecom companies: acquiring a new customer costs significantly more than retaining an existing one. The ability to **proactively identify at-risk customers** and support **data-driven retention strategies** is a direct business value driver.

This project demonstrates how a Data Analyst can bridge **exploratory data analysis**, **statistical modeling**, and **actionable business recommendations** to address this problem end-to-end.

---

## 🔗 Live Demo

👉 **https://telco-customer-churn-prediction-simonnc.streamlit.app/**

The deployed Streamlit application allows users to:

- Simulate individual customer profiles
- Estimate churn probability using the trained Random Forest model
- Adjust the decision threshold based on business strategy (cost of acquisition vs. retention)
- Receive actionable retention recommendations

---

## 🎯 Project Objective

Build an interpretable, production-ready binary classification model capable of predicting customer churn, with a **recall-first strategy** to minimize missed high-risk customers.

The focus is deliberately **business-oriented**: results are designed to support operational decisions, not to demonstrate academic ML techniques.

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| **Source** | Telco Customer Churn (IBM / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| **Granularity** | 1 row = 1 customer |
| **Size** | 7,043 customers |
| **Target** | `Churn` (Yes / No) |
| **Churn rate** | ~26.5% |

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (EDA)

Business-oriented analysis to identify key churn drivers before any modeling. The goal is to understand **why customers leave**, not just predict **who** will leave.

### 2. Feature Engineering

Business-driven feature selection, proper encoding, and **leakage-safe preprocessing pipelines** to ensure model reliability in production.

### 3. Modeling

| Model | ROC-AUC | Recall (Churn) | Precision (Churn) |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.80 | ~0.67 | ~0.52 |
| **Random Forest (final)** | **0.84** | **0.73** | 0.56 |

**Model selection rationale**: Random Forest was selected for its superior recall and discrimination, aligned with churn prevention objectives where missing a high-risk customer is costlier than a false alert.

### 4. Deployment

Final pipeline (preprocessing + model) exported and integrated into a Streamlit application ensuring prediction consistency with offline evaluation.

---

## 📈 Key Insights

| Finding | Business Implication |
|---|---|
| **Month-to-month contracts** show significantly higher churn | Contract commitment reduces churn risk |
| Churn risk peaks during the **first months** of the customer lifecycle | Onboarding and early engagement are critical |
| Absence of **Tech Support** strongly associated with higher churn | Service bundling as a retention lever |
| **Online Security** acts as a strong retention driver among internet customers | Targeted upsell opportunity |

These insights are **actionable without the model**: they inform product, pricing, and customer success strategies independently of ML predictions.

---

## 🗂️ Project Structure

```
telco-customer-churn-prediction/
├── data/
│   ├── raw/                       # Raw dataset
│   └── processed/                 # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb          # Model training & evaluation
├── src/
│   └── data_prep.py               # Reusable preprocessing
├── models/
│   └── churn_model_rf.joblib      # Trained model pipeline
├── app/
│   └── streamlit_app.py           # Decision-support app
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Run Locally

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install --upgrade pip
pip install -r requirements.txt

# Run notebooks
jupyter notebook

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## 🔍 Limitations & Next Steps

- Threshold tuning based on customer lifetime value and retention cost
- Feature importance analysis for deeper business interpretability
- Cost-sensitive or profit-based optimization
- Monitoring model performance over time (data drift detection)

---

## 🎯 Skills Demonstrated

This project demonstrates competencies aligned with **Data Analyst** market requirements:

| Competency | How it is demonstrated |
|---|---|
| **Exploratory data analysis (EDA)** | Business-oriented analysis of 7,043 customers to identify churn drivers |
| **Statistical modeling** | Logistic Regression baseline + Random Forest with recall-first strategy |
| **Python** (Pandas, Scikit-learn) | End-to-end pipeline from raw data to deployed model |
| **Data visualization** | Charts and findings designed to communicate with business stakeholders |
| **Business needs analysis** | Problem framed from the operational perspective (retention cost vs. acquisition cost) |
| **KPI design** | Churn rate, retention levers, and threshold tuning as decision-support metrics |
| **Deployment** | Streamlit app for real-time customer simulation and retention strategy testing |

---

## 🔗 Related Projects

This project complements the portfolio alongside:

- 👉 [Olist E-commerce: End-to-End BI Solution](https://github.com/SimonNC/olist-data-analysis) (Python + Power BI dashboards)
- 👉 [Olist Analytics Engineering Pipeline](https://github.com/SimonNC/olist-dbt-duckdb) (SQL + dbt)

---

## 👤 Author

**Simon Jorite**
Data Analyst - [Microsoft Certified Power BI Data Analyst (PL-300)](https://learn.microsoft.com/en-us/users/simonjorite-4846/credentials/b2cc3310a92a9302)

15 years of experience in finance, operations, and e-commerce. I transform complex datasets into reliable KPIs and decision-ready dashboards.

- GitHub: [github.com/SimonNC](https://github.com/SimonNC)
- LinkedIn: [linkedin.com/in/simonjorite](https://www.linkedin.com/in/simonjorite)
- Email: simon.jorite@gmail.com
- Location: Lyon, France (Open to hybrid / remote)
- Scheduling: [Book a 30-min exchange](https://calendly.com/simon-jorite/echange-da)
