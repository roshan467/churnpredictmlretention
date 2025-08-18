import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset (make sure dataset is inside data/customer_churn.csv)
df = pd.read_csv("data/customer_churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

# Features and target
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/churn_model.pkl")

print("✅ Model trained and saved at model/churn_model.pkl")
