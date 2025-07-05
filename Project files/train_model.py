import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


try:
    df = pd.read_csv("health_risk_data_balanced.csv")
except FileNotFoundError:
    raise FileNotFoundError("❌ File 'health_risk_data_balanced.csv' not found.")


required_cols = ["Heart Rate", "Blood Glucose", "Systolic BP", "Diastolic BP", "Sleep Hours", "Symptom", "Risk Level"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column: {col}")

df["Symptom"] = df["Symptom"].astype("category")
df["Symptom_Code"] = df["Symptom"].cat.codes


df["Risk Level"] = df["Risk Level"].astype("category")
label_map = dict(enumerate(df["Risk Level"].cat.categories))
df["Risk_Code"] = df["Risk Level"].cat.codes

joblib.dump(label_map, "label_map.joblib")


X = df[["Heart Rate", "Blood Glucose", "Systolic BP", "Diastolic BP", "Sleep Hours", "Symptom_Code"]]
y = df["Risk_Code"]


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)
joblib.dump(model, "risk_model.joblib")

print("✅ Model trained and saved as 'risk_model.joblib'")
