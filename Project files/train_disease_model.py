
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("disease_symptoms_binary.csv")

X = df.drop("disease", axis=1)
y = df["disease"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)


joblib.dump(model, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model and label encoder saved!")
