"""
Step 1: Load, clean, and explore the Telco Customer Churn dataset.
Business problem: Company knows customers are leaving, but doesn't know WHY
or WHO is most at risk right now.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---- 1. Load ----
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ---- 2. Clean ----
# TotalCharges is stored as text and has blank strings for brand-new customers (tenure=0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"Missing TotalCharges after conversion: {df['TotalCharges'].isna().sum()}")
df["TotalCharges"] = df["TotalCharges"].fillna(0)  # tenure=0 customers haven't been billed yet

# Drop ID column, not predictive
df = df.drop(columns=["customerID"])

# Target as binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- 3. Overall churn rate ----
churn_rate = df["Churn"].mean() * 100
print(f"Overall churn rate: {churn_rate:.1f}%")

# ---- 4. Churn by contract type ----
plt.figure(figsize=(6, 4))
contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
sns.barplot(x=contract_churn.index, y=contract_churn.values, palette="Reds_r")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")
plt.tight_layout()
plt.savefig(f"{OUT}/churn_by_contract.png", dpi=150)
plt.close()
print("Churn by contract:\n", contract_churn)

# ---- 5. Churn by tenure ----
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", palette=["#2ecc71", "#e74c3c"])
plt.title("Tenure Distribution by Churn Status")
plt.tight_layout()
plt.savefig(f"{OUT}/tenure_vs_churn.png", dpi=150)
plt.close()

# ---- 6. Monthly charges vs churn ----
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#2ecc71", "#e74c3c"])
plt.title("Monthly Charges by Churn Status")
plt.xticks([0, 1], ["Stayed", "Churned"])
plt.tight_layout()
plt.savefig(f"{OUT}/charges_vs_churn.png", dpi=150)
plt.close()

# ---- 7. Save cleaned data for the modeling step ----
df.to_csv(f"{OUT}/cleaned_data.csv", index=False)

print("\nEDA complete. Plots and cleaned_data.csv saved to outputs/")
