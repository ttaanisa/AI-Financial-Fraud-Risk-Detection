# ==============================
# AI Financial Fraud Detection & Risk Scoring System
# ==============================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Create folders if they do not exist
os.makedirs("Screenshots", exist_ok=True)
os.makedirs("Outputs", exist_ok=True)

# Clean professional chart style
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("data/creditcard.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nShape of dataset:", data.shape)
print("\nColumns:", data.columns)

# ==============================
# Class Distribution
# ==============================
print("\nClass Counts:")
print(data["Class"].value_counts())

print("\nClass Percentage:")
print(data["Class"].value_counts(normalize=True) * 100)

counts = data["Class"].value_counts(normalize=True).sort_index() * 100

plt.figure(figsize=(8, 5))
counts.plot(kind="bar")
plt.title("Fraud vs Normal Transactions (%)")
plt.xlabel("Transaction Class")
plt.ylabel("Percentage")
plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)

for i, value in enumerate(counts):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("Screenshots/fraud_vs_normal_percentage.png", dpi=300)
plt.show()

# ==============================
# Transaction Amount Distribution
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(data["Amount"], bins=50)
plt.yscale("log")
plt.title("Transaction Amount Distribution (Log Scale)")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency (Log Scale)")
plt.tight_layout()
plt.savefig("Screenshots/transaction_amount_distribution_log.png", dpi=300)
plt.show()

# ==============================
# Average Amount Comparison
# ==============================
normal_avg = data[data["Class"] == 0]["Amount"].mean()
fraud_avg = data[data["Class"] == 1]["Amount"].mean()

print("\nAverage amount (Normal):", normal_avg)
print("Average amount (Fraud):", fraud_avg)

# ==============================
# Boxplot Clean Version
# ==============================
plt.figure(figsize=(8, 5))
data.boxplot(column="Amount", by="Class", showfliers=False)
plt.title("Transaction Amount by Class (Outliers Removed)")
plt.suptitle("")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Amount")
plt.tight_layout()
plt.savefig("Screenshots/transaction_amount_by_class_boxplot.png", dpi=300)
plt.show()

# ==============================
# Train-Test Split
# ==============================
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ==============================
# Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Logistic Regression Basic
# ==============================
basic_model = LogisticRegression(max_iter=2000)
basic_model.fit(X_train_scaled, y_train)

y_pred_basic = basic_model.predict(X_test_scaled)

print("\nLogistic Regression (Basic)")
print(confusion_matrix(y_test, y_pred_basic))
print(classification_report(y_test, y_pred_basic))

# ==============================
# Logistic Regression Balanced
# ==============================
balanced_model = LogisticRegression(max_iter=2000, class_weight="balanced")
balanced_model.fit(X_train_scaled, y_train)

y_pred_balanced = balanced_model.predict(X_test_scaled)

print("\nLogistic Regression (Balanced)")
print(confusion_matrix(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced))

# ==============================
# Random Forest Model
# ==============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

print("\nRandom Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ==============================
# Confusion Matrix Visualization
# ==============================
cm = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("Screenshots/confusion_matrix_random_forest.png", dpi=300)
plt.show()

# ==============================
# Fraud Probability and Risk Scoring
# ==============================
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred_rf
results["Fraud_Probability"] = y_prob

def risk_level(prob):
    if prob > 0.8:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

results["Risk_Level"] = results["Fraud_Probability"].apply(risk_level)

print("\nSample Results:")
print(results[["Actual", "Predicted", "Fraud_Probability", "Risk_Level"]].head())

# Save results
results.to_csv("Outputs/fraud_risk_predictions.csv", index=False)

# ==============================
# Fraud Probability Distribution
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(results["Fraud_Probability"], bins=50)
plt.title("Fraud Probability Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("Screenshots/fraud_probability_distribution.png", dpi=300)
plt.show()

# ==============================
# Model Comparison Summary
# ==============================
print("\nProject Complete.")
print("Screenshots saved in: Screenshots/")
print("Prediction results saved in: Outputs/fraud_risk_predictions.csv")