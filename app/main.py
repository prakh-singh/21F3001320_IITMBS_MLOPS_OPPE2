from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and feature list at startup
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode
    df = pd.get_dummies(df)

    # Align columns with training features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

