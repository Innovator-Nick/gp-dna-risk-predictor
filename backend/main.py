from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="GP DNA Predictor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
print("🔄 Loading model...")
try:
    model = joblib.load('../model/dna_model.pkl')
    feature_columns = joblib.load('../model/feature_columns.pkl')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    feature_columns = None


class PredictionRequest(BaseModel):
    hcp_type: str
    appt_mode: str
    age: int
    hour: int
    day_of_week: int
    lead_time: int


@app.get("/")
def root():
    return {
        "message": "GP DNA Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_dna(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Create dataframe
        data = pd.DataFrame([{
            'hcp_type': request.hcp_type,
            'appt_mode': request.appt_mode,
            'age': request.age,
            'hour': request.hour,
            'day_of_week': request.day_of_week,
            'lead_time': request.lead_time
        }])

        # Encode
        data_encoded = pd.get_dummies(data, columns=['hcp_type', 'appt_mode'])

        # Align features
        for col in feature_columns:
            if col not in data_encoded.columns:
                data_encoded[col] = 0

        data_encoded = data_encoded[feature_columns]

        # Predict
        probability = float(model.predict_proba(data_encoded)[0][1])
        prediction = int(model.predict(data_encoded)[0])

        # Risk level
        if probability < 0.08:
            risk_level = "Low"
            color = "green"
            recommendation = "Standard reminder process sufficient"
        elif probability < 0.15:
            risk_level = "Medium"
            color = "orange"
            recommendation = "Send confirmation SMS 24 hours before"
        else:
            risk_level = "High"
            color = "red"
            recommendation = "Call patient to confirm + send SMS"

        return {
            "prediction": prediction,
            "probability": round(probability, 3),
            "risk_level": risk_level,
            "color": color,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats")
def get_stats():
    return {
        "model_type": "Gradient Boosting Classifier",
        "accuracy": "91-93%",
        "training_size": "10,000 NHS appointments"
    }