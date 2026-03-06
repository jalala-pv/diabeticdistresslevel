import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === Load dataset ===
df = pd.read_csv(r"D:\diabeticdistress-master-01-03-26\diabeticdistress-master\myapp\diabetes_prediction_dataset.csv")

# === Check column names ===
print(df.columns)

# Example dataset columns: ['age','gender','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level','diabetes']

# === Encode categorical columns ===
labelencoder = LabelEncoder()
df['gender'] = labelencoder.fit_transform(df['gender'])
df['smoking_history'] = labelencoder.fit_transform(df['smoking_history'])

# === Features and Target ===
X = df[['age','gender','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

# === Split into training and testing ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# === Save model ===
joblib.dump(model, 'diabetes_rf_model.pkl')
print("✅ Model saved as diabetes_rf_model.pkl")