"""
DiaBD Diabetes Risk API

Endpoints
  GET  /api/health    -> is the API up, which model is loaded
  GET  /api/features  -> expected inputs, units and valid ranges
  POST /api/predict   -> diabetes risk estimate for one person

Run:
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000  (the API also serves ../frontend/index.html)
Interactive docs: http://localhost:8000/docs
"""
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"
FRONTEND_DIR = HERE.parent / "frontend"

if not MODEL_PATH.exists():
    raise SystemExit("model.joblib not found. Run `python train_model.py` first.")

bundle = joblib.load(MODEL_PATH)
model, scaler = bundle["model"], bundle["scaler"]
FEATURES: list[str] = bundle["features"]
MEANS: dict[str, float] = bundle["means"]

# Same cut-offs the website already uses for its Low / Moderate / High labels
MODERATE_AT, HIGH_AT = 35.0, 70.0

app = FastAPI(title="DiaBD Diabetes Risk API", version="1.0.0")

# Only needed if the page is hosted on a different origin than the API
# (e.g. Live Server on :5500 calling the API on :8000). Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    """All fields are optional: anything left out is filled with the dataset mean,
    exactly like the website does. Units are the model's units
    (glucose in mmol/L, height in metres)."""

    age: Optional[float] = Field(None, ge=18, le=120, description="years")
    gender: Optional[Literal[0, 1]] = Field(None, description="0 = female, 1 = male")
    pulse_rate: Optional[float] = Field(None, ge=30, le=150, description="beats/min")
    systolic_bp: Optional[float] = Field(None, ge=60, le=250, description="mm Hg")
    diastolic_bp: Optional[float] = Field(None, ge=10, le=130, description="mm Hg")
    glucose: Optional[float] = Field(None, ge=0.05, le=35, description="mmol/L")
    height: Optional[float] = Field(None, ge=0.5, le=2.5, description="metres")
    weight: Optional[float] = Field(None, ge=10, le=200, description="kg")
    bmi: Optional[float] = Field(None, ge=10, le=80, description="kg/m²")
    family_diabetes: Optional[Literal[0, 1]] = None
    hypertensive: Optional[Literal[0, 1]] = None
    family_hypertension: Optional[Literal[0, 1]] = None
    cardiovascular_disease: Optional[Literal[0, 1]] = None
    stroke: Optional[Literal[0, 1]] = None


class Contribution(BaseModel):
    feature: str
    contribution: float  # change in log-odds vs. an average person


class PredictionOutput(BaseModel):
    risk_percent: float
    band: Literal["Low", "Moderate", "High"]
    inputs_provided: int
    inputs_total: int
    contributions: list[Contribution]  # one per feature, in model order
    model_info: dict


def band_for(p: float) -> str:
    return "High" if p >= HIGH_AT else "Moderate" if p >= MODERATE_AT else "Low"


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "logistic_regression", "metrics": bundle["metrics"]}


@app.get("/api/features")
def features():
    out = {}
    for name, field in PredictionInput.model_fields.items():
        ge = next((m.ge for m in field.metadata if hasattr(m, "ge")), None)
        le = next((m.le for m in field.metadata if hasattr(m, "le")), None)
        out[name] = {"min": ge, "max": le, "unit": field.description, "dataset_mean": round(MEANS[name], 4)}
    return out


@app.post("/api/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    data = payload.model_dump()
    provided = sum(v is not None for v in data.values())
    if provided == 0:
        raise HTTPException(status_code=400, detail="Provide at least one input.")

    row = pd.DataFrame([[data[f] if data[f] is not None else MEANS[f] for f in FEATURES]], columns=FEATURES)
    scaled = scaler.transform(row)
    p = float(model.predict_proba(scaled)[0, 1]) * 100
    contrib = scaled[0] * model.coef_[0]

    return PredictionOutput(
        risk_percent=round(p, 3),
        band=band_for(p),
        inputs_provided=provided,
        inputs_total=len(FEATURES),
        contributions=[Contribution(feature=f, contribution=round(float(c), 5)) for f, c in zip(FEATURES, contrib)],
        model_info={"type": "logistic_regression", **bundle["metrics"]},
    )


# Serve the website from the same server (registered last so /api/* wins).
# Same origin = no CORS headaches and one command to run everything.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
