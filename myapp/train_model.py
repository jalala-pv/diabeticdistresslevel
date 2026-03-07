import pandas as pd
import numpy as np
# RandomForestClassifier Machine learning algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Shows model performance
from sklearn.metrics import classification_report
# Saves trained model
import joblib

# Load the dataset using a raw string for the Windows path
path = r'D:\diabeticdistress-master-01-03-26\diabeticdistress-master\myapp\randomforest_dataset.csv'
df = pd.read_csv(path)

# 21 Features in the EXACT order shown in your dataset screenshot
features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

X = df[features]
y = df['Diabetes_012'] # Target: 0=Low, 1=Moderate, 2=High

# Split the data random_state=42 ensures same split every time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training with Optimizations:
# 1. n_estimators=200: More trees usually lead to more stable predictions.
# 2. class_weight='balanced': Helps if you have more 'Low' cases than 'High' cases.
# 3. random_state=42: Ensures you get the same results every time you train.
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
# This step trains the model.
model.fit(X_train, y_train)

# VERIFICATION: Print a report to see how well it's actually doing
y_pred = model.predict(X_test)
# Precision Correct positive predictions
# Recall How many real cases detected
# F1-score Balance between precision & recall
# Support Number of samples
print("--- Training Performance Report ---")
print(classification_report(y_test, y_pred))

# Save the model
save_path = r'D:\diabeticdistress-master-01-03-26\diabeticdistress-master\myapp\diabetes_rf_model1.pkl'
joblib.dump(model, save_path)
print(f"Model saved successfully to {save_path}")