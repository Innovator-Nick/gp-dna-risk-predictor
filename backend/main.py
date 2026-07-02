from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import os

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="GP DNA Risk Prediction API",
    description="Predicts GP appointment no-show (DNA) risk using Gradient Boosting",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Paths ───────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "../model/dna_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../model/feature_columns.pkl")
THRESHOLD_PATH= os.path.join(BASE_DIR, "../model/threshold.pkl")

model        = None
feature_cols = None
THRESHOLD    = 0.35  # fallback if threshold.pkl missing

# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, feature_cols, THRESHOLD
    try:
        model        = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        print(f"[OK] Model loaded — {len(feature_cols)} features")
    except FileNotFoundError as e:
        print(f"[ERROR] Model file missing: {e}")
        print("Run backend/train_model.py first.")
    try:
        THRESHOLD = float(joblib.load(THRESHOLD_PATH))
        print(f"[OK] Threshold loaded — {THRESHOLD}")
    except FileNotFoundError:
        print(f"[WARN] threshold.pkl not found — using default {THRESHOLD}")

# ── Valid values (used for validation + /meta) ────────────────
VALID_HCP   = ["GP", "Nurse", "Other Practice staff", "Mental Health"]
VALID_MODE  = ["Face-to-face", "Telephone", "Video/Online"]
VALID_DAYS  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# ── Request / Response Schemas ─────────────────────────────────
class AppointmentRequest(BaseModel):
    hcp_type:          str = Field(..., description="GP | Nurse | Other Practice staff | Mental Health")
    appt_mode:         str = Field(..., description="Face-to-face | Telephone | Video/Online")
    age:               int = Field(..., ge=0,  le=120, description="Patient age in years")
    hour:              int = Field(..., ge=8,  le=17,  description="Appointment hour (8–17)")
    day_of_week:       int = Field(..., ge=0,  le=4,   description="0=Monday … 4=Friday")
    lead_time:         int = Field(..., ge=0,  le=60,  description="Days between booking and appointment")
    prior_dna_count:   int = Field(..., ge=0,  le=5,   description="Number of DNAs in the past 12 months (0–5)")
    deprivation_score: int = Field(..., ge=1,  le=10,  description="IMD deprivation score 1–10 (10 = most deprived)")

    @validator("hcp_type")
    def validate_hcp(cls, v):
        if v not in VALID_HCP:
            raise ValueError(f"hcp_type must be one of {VALID_HCP}")
        return v

    @validator("appt_mode")
    def validate_mode(cls, v):
        if v not in VALID_MODE:
            raise ValueError(f"appt_mode must be one of {VALID_MODE}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "hcp_type":          "GP",
                "appt_mode":         "Face-to-face",
                "age":               28,
                "hour":              9,
                "day_of_week":       0,
                "lead_time":         21,
                "prior_dna_count":   1,
                "deprivation_score": 6,
            }
        }

class PredictionResponse(BaseModel):
    prediction:     int
    probability:    float
    risk_level:     str
    color:          str
    recommendation: str

# ── Helpers ───────────────────────────────────────────────────
def _risk_band(prob: float) -> tuple[str, str, str]:
    """Return (risk_level, colour, recommendation) for a given probability."""
    if prob < 0.20:
        return (
            "Low",
            "green",
            "Standard 48h SMS reminder is sufficient.",
        )
    elif prob < 0.35:
        return (
            "Medium",
            "orange",
            "Send SMS reminder and consider a phone confirmation call.",
        )
    else:
        return (
            "High",
            "red",
            "Call patient directly 24h before and send SMS reminder.",
        )

def _build_features(req: AppointmentRequest) -> pd.DataFrame:
    """Convert request to model-ready DataFrame aligned to training feature columns."""
    raw = {
        "hcp_type":           req.hcp_type,
        "appt_mode":          req.appt_mode,
        "age":                req.age,
        "hour":               req.hour,
        "day_of_week":        req.day_of_week,
        "lead_time":          req.lead_time,
        "prior_dna_count":    req.prior_dna_count,
        "deprivation_score":  req.deprivation_score,
    }
    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=["hcp_type", "appt_mode"])

    # Add any columns present in training but missing here (set to 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols]  # enforce exact column order

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "message":      "GP DNA Risk Prediction API",
        "status":       "running",
        "version":      "2.0.0",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict DNA risk")
def predict(request: AppointmentRequest):
    """
    Predict the probability that a GP appointment will be a Did Not Attend (DNA).

    Returns a risk level (Low / Medium / High), probability score, and a
    plain-English recommendation for practice staff.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py and restart the server.",
        )

    df   = _build_features(request)
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= THRESHOLD)
    risk, colour, rec = _risk_band(prob)

    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 4),
        risk_level=risk,
        color=colour,
        recommendation=rec,
    )


@app.get("/stats", summary="Model statistics")
def stats():
    """Return key model performance metrics."""
    return {
        "model_type":       "GradientBoostingClassifier (scikit-learn)",
        "roc_auc":          "0.70",
        "test_accuracy":    "83.2%",
        "training_accuracy":"86.9%",
        "precision_dna":    "0.32",
        "recall_dna":       "0.43",
        "f1_score_dna":     "0.37",
        "decision_threshold": THRESHOLD,
        "features":         len(feature_cols) if feature_cols else "model not loaded",
        "training_size":    "10,000 synthetic appointments (NHS England OGL data patterns)",
        "note":             "ROC-AUC is the primary metric; accuracy alone is misleading for imbalanced classes",
    }


@app.get("/meta", summary="Valid input values")
def meta():
    """Return valid input enumerations for frontend dropdowns."""
    return {
        "hcp_types":    VALID_HCP,
        "appt_modes":   VALID_MODE,
        "days_of_week": VALID_DAYS,
        "age_range":           {"min": 0,  "max": 120},
        "hour_range":          {"min": 8,  "max": 17},
        "lead_time_range":     {"min": 0,  "max": 60},
        "prior_dna_range":     {"min": 0,  "max": 5},
        "deprivation_range":   {"min": 1,  "max": 10},
    }
