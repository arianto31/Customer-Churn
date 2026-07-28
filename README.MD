# Bank Customer Churn Prediction

This project aims to predict customer churn for a banking institution using machine learning. By identifying customers likely to leave, the bank can proactively implement retention strategies.

## 📊 Project Overview

In the banking industry, retaining customers is more cost-effective than acquiring new ones. This project involves a comprehensive data science workflow, from exploratory data analysis to model interpretation, to build a robust churn prediction system.

### Key Objectives:
- Identify primary drivers of customer churn.
- Build a predictive model capable of handling imbalanced data.
- Provide interpretable results for business stakeholders using SHAP values.

## 📂 Project Structure

```text
6customer_churn/
├── data/
│   ├── Bank Customer Churn Prediction.csv  # Raw dataset
│   └── cleaned_engineered_churn.csv        # Processed data
├── main.ipynb                              # Core analysis & modeling notebook
└── README.md                               # Project documentation
```

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:**
  - **Data Analysis:** `pandas`, `numpy`
  - **Visualization:** `matplotlib`, `seaborn`
  - **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
  - **Interpretability:** `shap`

## 📈 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed feature distributions and correlation matrices.
- Identified class imbalance: ~80% stayed vs. ~20% churned.
- Detected and handled outliers using the IQR method.

### 2. Data Preprocessing & Feature Engineering
- **Numerical Features:** Normalized using `StandardScaler`.
- **Categorical Features:** Transformed using `OneHotEncoder`.
- **Pipelines:** Integrated preprocessing and modeling into Scikit-learn `Pipeline` for reproducibility and to prevent data leakage.

### 3. Handling Class Imbalance
Compared two main approaches:
- **SMOTE (Synthetic Minority Over-sampling Technique):** Creating synthetic examples of the churned class.
- **Class Weights:** Penalizing the model more for misclassifying the minority class.

### 4. Modeling & Hyperparameter Tuning
- Evaluated multiple models: Logistic Regression, Random Forest, XGBoost, and LightGBM.
- Optimized performance using `RandomizedSearchCV` with Stratified K-Fold cross-validation.

### 5. Evaluation
- Used metrics suited for imbalanced data: **F1-score**, **ROC-AUC**, and **PR-AUC (Average Precision)**.
- Analyzed Confusion Matrices to understand False Positives vs. False Negatives.

## 🔍 Key Insights (Model Interpretation)

Using SHAP (SHapley Additive exPlanations), we identified the most influential factors:
1. **Age:** Older customers are significantly more likely to churn.
2. **Number of Products:** Customers with a specific number of products (e.g., 3 or 4) show a very high churn rate.
3. **Is Active Member:** Non-active members have a higher propensity to leave.

## 🚀 How to Run

1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may need to create a requirements.txt based on the imports in main.ipynb)*
3. Open `main.ipynb` in VS Code or Jupyter Notebook.
4. Run all cells to reproduce the analysis.