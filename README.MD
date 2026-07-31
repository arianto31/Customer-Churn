# 🏦 Bank Customer Churn Prediction & Streamlit Dashboard

An end-to-end machine learning project for predicting customer churn in the banking industry. This project covers the complete data science workflow—from exploratory data analysis (EDA), data preprocessing, feature engineering, model development, hyperparameter tuning, model interpretation, to deployment using Streamlit for real-time customer churn prediction.

🌐 **Streamlit Dashboard:**  
https://bank-customer-churn-project.streamlit.app/

---

## 📖 Project Overview

Customer churn is one of the biggest challenges in the banking industry. Losing existing customers not only reduces revenue but also increases customer acquisition costs. This project leverages machine learning to identify customers who are at risk of leaving, enabling banks to implement proactive retention strategies.

The project combines data analytics, predictive modelling, explainable AI, and web application development into a single solution.

### Objectives

- Predict whether a customer will churn.
- Identify the most influential factors affecting churn.
- Handle imbalanced datasets effectively.
- Provide interpretable predictions using SHAP.
- Deploy an interactive dashboard for business users.

---

## 🚀 Features

- 📊 Comprehensive Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Feature Engineering
- ⚖️ Imbalanced Data Handling (SMOTE & Class Weight)
- 🤖 Multiple Machine Learning Models
- 🔍 Hyperparameter Optimisation
- 📈 Model Performance Evaluation
- 💡 Explainable AI with SHAP
- 🌐 Interactive Streamlit Dashboard
- ⚡ Real-time Customer Churn Prediction

---

## 📂 Project Structure

```text
churn-project/
│
├── data/
│   ├── Bank Customer Churn Prediction.csv
│   └── cleaned_engineered_churn.csv
│
├── main.ipynb
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Web Framework

- Streamlit

### Data Processing

- Pandas
- NumPy

### Data Visualisation

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn

### Explainable AI

- SHAP

### Model Persistence

- Joblib

---

# 📊 Dataset

The dataset contains customer information collected from a banking institution.

Typical features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Target Variable:

- **Exited**
  - 0 = Customer Stayed
  - 1 = Customer Churned

---

# 🔬 Machine Learning Workflow

## 1. Exploratory Data Analysis (EDA)

The first stage focused on understanding the characteristics of the dataset.

Performed analyses include:

- Missing value inspection
- Duplicate checking
- Feature distribution analysis
- Correlation analysis
- Boxplots for outlier detection
- Customer churn distribution
- Numerical and categorical feature analysis

### Findings

- The dataset is highly imbalanced.
- Approximately **80%** of customers stayed.
- Approximately **20%** of customers churned.
- Several numerical variables contain outliers.

---

## 2. Data Preprocessing

The preprocessing pipeline includes:

### Numerical Features

- Missing value handling
- StandardScaler normalisation

### Categorical Features

- One-Hot Encoding

### Feature Engineering

Additional engineered features include customer segmentation such as:

- Age Group
- Credit Score Category

The preprocessing workflow is implemented using **Scikit-learn Pipelines**, ensuring reproducibility and preventing data leakage.

---

## 3. Handling Imbalanced Data

Since customer churn represents the minority class, two strategies were evaluated:

### SMOTE

Synthetic Minority Over-sampling Technique generates synthetic churn samples to balance the dataset.

### Class Weight

Instead of generating new samples, the model penalises misclassification of minority-class customers.

Both approaches were compared to determine the best-performing solution.

---

## 4. Model Development

Multiple classification algorithms were trained and compared.

Models include:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Each model was evaluated using the same preprocessing pipeline.

---

## 5. Hyperparameter Tuning

To improve predictive performance, hyperparameter optimisation was performed using:

- RandomizedSearchCV
- Stratified K-Fold Cross Validation

This ensures robust model evaluation while preserving class distribution across folds.

---

## 6. Model Evaluation

Because the dataset is imbalanced, evaluation focused on metrics beyond accuracy.

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Average Precision)

Additional analyses include:

- Confusion Matrix
- Precision-Recall Curve
- ROC Curve

---

# 🔍 Model Interpretation

To improve model transparency and business understanding, SHAP (SHapley Additive exPlanations) was used.

The most influential features include:

- **Age**
  - Older customers are more likely to churn.

- **Number of Products**
  - Customers owning three or four products exhibit a significantly higher churn probability.

- **Is Active Member**
  - Inactive customers are much more likely to leave the bank.

- **Balance**
  - Higher account balances contribute to churn risk for certain customer groups.

- **Credit Score**
  - Lower credit scores slightly increase churn probability.

SHAP enables both global and individual prediction explanations.

---

# 🌐 Streamlit Dashboard

🔗 **Live Application:**  
https://bank-customer-churn-project.streamlit.app/

The project includes an interactive Streamlit application for real-time prediction.

### Dashboard Features

- Customer data input form
- Real-time churn prediction
- Churn probability score
- Interactive visualisations
- Model insights
- SHAP-based feature interpretation

---

# 📈 Results

The final model successfully identifies customers at high risk of churn while addressing class imbalance through appropriate evaluation metrics and optimisation techniques.

Business users can leverage the deployed application to:

- Identify high-risk customers
- Support customer retention strategies
- Improve marketing campaigns
- Assist decision-making through explainable AI

---

# ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/churn-project.git
cd churn-project
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

### Or Try the Live Demo

🌐 https://bank-customer-churn-project.streamlit.app/

---

# 📚 Libraries Used

```text
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
joblib
streamlit
```

---

# 👨‍💻 Author

**Arianto Blawa Maran**

Mathematics Graduate | Data Scientist | Machine Learning Engineer

- Python
- SQL
- Machine Learning
- Data Analytics
- Explainable AI (SHAP)
- Streamlit
- XGBoost
- LightGBM
- Scikit-learn

---

## ⭐ If you found this project helpful, consider giving it a star!