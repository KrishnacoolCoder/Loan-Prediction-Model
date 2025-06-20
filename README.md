# Loan-Prediction-Model

Loan Approval Prediction using XGBoost

This project focuses on building a high-performance classification model using XGBoost to predict loan approval decisions based on customer financial and demographic data.

Objective
To accurately classify loan applications as "Approved" or "Rejected", minimizing false approvals (default risk) while maintaining fairness.

Dataset
A preprocessed loan application dataset with features including income, employment history, credit score, loan amount, etc.

Tools & Technologies
- Python
- XGBoost
- Scikit-learn
- Pandas, NumPy, Seaborn, Matplotlib

Key Steps
- Performed EDA and handled missing values/outliers
- Trained XGBoost Classifier with hyperparameter tuning
- Optimized threshold using F-beta (β = 0.5) to emphasize precision
- Evaluated model with F1 score, confusion matrix, precision-recall tradeoff

Results
- **Accuracy:** 95%
- **F1-Score (Approved class):** 0.96
- **False Approvals:** Only 3 out of 614
- **Confusion Matrix & Classification Report** included

