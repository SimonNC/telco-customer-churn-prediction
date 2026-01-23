# Telco Customer Churn Prediction

## 📌 Business Context
Customer churn is a critical challenge for telecom companies, as acquiring new customers is significantly more expensive than retaining existing ones.

This project aims to predict customer churn in order to identify at-risk customers and support data-driven retention strategies.

---

## 🎯 Project Objective
Build a binary classification model capable of predicting whether a customer is likely to churn, with the goal of prioritizing retention actions and reducing the overall churn rate.

---

## 📊 Dataset
- **Source**: Telco Customer Churn dataset (IBM / Kaggle)
- **Granularity**: One row per customer
- **Target variable**: `Churn` (Yes / No)

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

### 3) Download the dataset

The dataset is downloaded programmatically using `kagglehub` in the EDA notebook and copied to `data/raw/`.

If needed:

```bash
pip install kagglehub
```

---

### 4) Run the notebooks

```bash
jupyter notebook
```

---

## 🧠 Approach

- Business understanding and problem framing  
- Exploratory Data Analysis (EDA) focused on churn drivers  
- Feature engineering based on business insights  
- Baseline and improved classification models  
- Model evaluation using churn-oriented metrics  
- Simple Streamlit application for customer scoring  

---

## 📈 Key Insights (EDA)

- Customers on month-to-month contracts exhibit significantly higher churn rates than those on one- or two-year contracts.
- Churn risk is highest during the early months of the customer lifecycle.
- The absence of Tech Support is associated with a substantially higher churn rate.
- Among internet customers, Online Security appears to be a strong retention driver.

---

## 🚀 Application

A lightweight Streamlit application will allow users to estimate churn risk for individual customers.

---

## 🔍 Limitations & Next Steps

- Further feature engineering and modeling are required to quantify the relative importance of churn drivers.
- Model performance will be evaluated using churn-oriented metrics (recall, precision, ROC-AUC).
- Future improvements may include cost-sensitive modeling and threshold optimization.
