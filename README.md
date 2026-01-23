# Telco Customer Churn Prediction

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://telco-customer-churn-prediction-simonnc.streamlit.app/)

## 📌 Business Context
Customer churn is a critical challenge for telecom companies, as acquiring new customers is significantly more expensive than retaining existing ones.

This project focuses on predicting customer churn in order to proactively identify at-risk customers and support data-driven retention strategies.

---

## 🔗 Live Demo

👉 **https://telco-customer-churn-prediction-simonnc.streamlit.app/**

The deployed Streamlit application allows users to:
- Simulate individual customer profiles
- Estimate churn probability using a trained Random Forest model
- Adjust the decision threshold based on business strategy
- Receive actionable retention recommendations

---

## 🎯 Project Objective
Build an interpretable and production-ready binary classification model capable of predicting whether a customer is likely to churn, with a strong focus on **recall** to avoid missing high-risk customers.

---

## 📊 Dataset
- **Source**: Telco Customer Churn dataset (IBM / [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Granularity**: One row per customer
- **Target variable**: `Churn` (Yes / No)
- **Churn rate**: ~26.5%

---

## 🗂️ Project Structure

```
telco-customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── data_prep.py
├── models/
│   └── churn_model_rf.joblib
├── app/
│   └── streamlit_app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Getting Started

### 1) Create and activate a virtual environment
**Windows (Git Bash)**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

---

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3) Dataset acquisition
The dataset is downloaded programmatically from Kaggle using `kagglehub` and stored locally in `data/raw/` to ensure reproducibility.

```bash
pip install kagglehub
```

---

### 4) Run notebooks
```bash
jupyter notebook
```

---

## 🧠 Methodology

1. **Exploratory Data Analysis (EDA)**  
   Identify key churn drivers from a business perspective.

2. **Feature Engineering**  
   Business-driven feature selection, proper encoding, and leakage-safe preprocessing pipelines.

3. **Modeling**
   - Baseline: Logistic Regression (interpretable reference)
   - Improved model: Random Forest (non-linear interactions)

4. **Evaluation**
   Metrics aligned with churn prevention use cases:
   - ROC-AUC
   - Recall on churners
   - Confusion matrix analysis

5. **Deployment**
   Export of the final pipeline and integration into a Streamlit application.

---

## 📈 Key Insights (EDA)

- Customers on **month-to-month contracts** show significantly higher churn rates.
- Churn risk is highest during the **early months** of the customer lifecycle.
- The absence of **Tech Support** is strongly associated with higher churn.
- Among internet customers, **Online Security** acts as a strong retention lever.

---

## 🤖 Modeling Results

| Model | ROC-AUC | Recall (Churn) | Precision (Churn) |
|------|--------|----------------|-------------------|
| Logistic Regression (baseline) | ~0.80 | ~0.67 | ~0.52 |
| Random Forest (final) | **0.84** | **0.73** | 0.56 |

**Model selection rationale:**  
The Random Forest model was selected as the final model due to its superior recall and overall discrimination, which better aligns with churn prevention objectives.

---

## 🚀 Streamlit Application

The Streamlit app is connected directly to the trained pipeline (preprocessing + model) to ensure prediction consistency with the offline evaluation.

🔗 **Live demo**: https://telco-customer-churn-prediction-simonnc.streamlit.app/

### Run the app locally
```bash
streamlit run app/streamlit_app.py
```

---

## 🔍 Limitations & Next Steps

- Threshold tuning based on customer lifetime value and retention cost
- Feature importance analysis for deeper business interpretability
- Cost-sensitive or profit-based optimization
- Monitoring model performance over time (data drift)

---

## 🧠 What This Project Demonstrates

- Ability to translate a **business problem into a data science pipeline**
- Strong focus on **interpretability and decision-making**
- Clean project structure and reusable code
- End-to-end workflow from EDA to deployment
