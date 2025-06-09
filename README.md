# Credit Risk Scoring Model

This project builds a machine learning model to predict credit risk (default vs. non-default) using the German Credit Dataset. It applies logistic regression with explainability tools (SHAP) to support financial decision-making and potential deployment in pre-approval credit pipelines.

## 🔍 Objective

- Predict whether a borrower poses a **high credit risk** (default) or **low risk** (non-default)
- Gain interpretable insights into **what drives risk decisions**, using SHAP values
- Lay the groundwork for **scalable deployment** in credit scoring systems

---

## 📊 Dataset

- **Source**: UCI German Credit Dataset  
- **Rows**: 1,000 customers  
- **Target**: `credit_risk` (1 = Bad, 0 = Good)
- **Features**: Loan amount, duration, credit history, age, employment, purpose, savings, etc.

---

## 🧠 Model Overview

- **Model**: Logistic Regression (baseline)
- **Evaluation Metrics**:
  - Accuracy: 78%
  - ROC AUC: 0.82
  - Recall for risky applicants: 90%
- **Interpretability**: SHAP used to explain key model decisions

---

## 📈 Key Insights

- Higher loan amounts and no checking account status are strong indicators of default risk
- Model recall was prioritized to reduce false negatives (missed risky borrowers)
- SHAP plots highlight the top features influencing each prediction

---

## 🧰 Tech Stack

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- SHAP (for model explainability)
- Jupyter Notebook
