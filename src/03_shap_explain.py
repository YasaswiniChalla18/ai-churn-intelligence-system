"""
Step 3: Explain WHY the model predicts churn - not just that it does.
This is the piece that separates this project from a generic "I trained a model" submission.
"""
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

OUT = "outputs"
rf = joblib.load(f"{OUT}/rf_model.joblib")
X_test = pd.read_csv(f"{OUT}/X_test.csv")

# ---- 1. Build SHAP explainer for the tree model ----
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# shap_values shape depends on sklearn/shap version - handle both binary output formats
if isinstance(shap_values, list):
    sv_churn = shap_values[1]  # class 1 = churned
else:
    sv_churn = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

# ---- 2. Global summary: which features matter most overall ----
plt.figure()
shap.summary_plot(sv_churn, X_test, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary_global.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- 3. Local explanation: pick one high-risk customer and explain THEIR prediction ----
proba = rf.predict_proba(X_test)[:, 1]
X_test_reset = X_test.reset_index(drop=True)
top_risk_idx = proba.argmax()
print(f"Highest-risk customer in test set: row {top_risk_idx}, "
      f"predicted churn probability = {proba[top_risk_idx]:.2f}")

# Top contributing features for this one customer
contrib = pd.Series(sv_churn[top_risk_idx], index=X_test.columns).sort_values(key=abs, ascending=False)
print("\nTop 5 factors driving this customer's churn risk:")
print(contrib.head(5))

ev = explainer.expected_value
base_value = float(ev[1]) if hasattr(ev, "__len__") and len(ev) > 1 else float(ev)
explanation = shap.Explanation(
    values=sv_churn[top_risk_idx],
    base_values=base_value,
    data=X_test_reset.iloc[top_risk_idx],
    feature_names=X_test.columns.tolist(),
)
plt.figure()
shap.plots.waterfall(explanation, max_display=10, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_local_example.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSHAP plots saved: shap_summary_global.png, shap_local_example.png")
