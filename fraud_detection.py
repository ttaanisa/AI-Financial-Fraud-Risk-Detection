# Import libraries
import pandas as pd

# Load dataset
data = pd.read_csv("data/creditcard.csv")

# Display first 5 rows
print(data.head())

# Show dataset shape (rows, columns)
print("Shape of dataset:", data.shape)

# Show column names
print("Columns:", data.columns)

# Count fraud vs normal transactions
print(data['Class'].value_counts())

# Show percentage distribution
print(data['Class'].value_counts(normalize=True))

import matplotlib.pyplot as plt

# Plot fraud vs normal transactions
data['Class'].value_counts().plot(kind='bar')

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.show()

# Plot transaction amounts
plt.hist(data['Amount'], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()

# Compare average transaction amounts
print("Average amount (Normal):", data[data['Class'] == 0]['Amount'].mean())
print("Average amount (Fraud):", data[data['Class'] == 1]['Amount'].mean())

# Boxplot comparison
import matplotlib.pyplot as plt

data.boxplot(column='Amount', by='Class')

plt.title("Transaction Amount by Class")
plt.suptitle("")  # removes extra title
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Amount")

plt.show()

from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))

model = LogisticRegression(max_iter=1000, class_weight='balanced')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Get probability scores
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Add to dataframe
results = X_test.copy()
results = pd.DataFrame(results)

results['Actual'] = y_test.values
results['Predicted'] = y_pred_rf
results['Fraud_Probability'] = y_prob

print(results.head())

# Create risk levels
def risk_level(prob):
    if prob > 0.8:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

results['Risk_Level'] = results['Fraud_Probability'].apply(risk_level)

print(results[['Fraud_Probability', 'Risk_Level']].head())