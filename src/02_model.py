"""
Step 2: Predict churn probability.
We compare Logistic Regression (interpretable baseline) vs Random Forest
(usually stronger, still explainable via SHAP later).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import os

OUT = "outputs"
df = pd.read_csv(f"{OUT}/cleaned_data.csv")

# ---- 1. Encode categorical columns ----
cat_cols = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

# ---- 2. Train/test split (stratified, since churn is imbalanced ~26%/74%) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Churn rate in train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ---- 3. Baseline: Logistic Regression ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)
log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, log_preds, target_names=["Stayed", "Churned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, log_proba):.3f}")

# ---- 4. Random Forest ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(classification_report(y_test, rf_preds, target_names=["Stayed", "Churned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")

# ---- 5. Confusion matrix for the better model (Random Forest) ----
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
disp.plot(cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# ---- 6. Save everything needed for the SHAP step ----
joblib.dump(rf, f"{OUT}/rf_model.joblib")
X_test.to_csv(f"{OUT}/X_test.csv", index=False)
y_test.to_csv(f"{OUT}/y_test.csv", index=False)
X_train.to_csv(f"{OUT}/X_train.csv", index=False)

print("\nModel + test data saved to outputs/ for the SHAP explainability step.")
