
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "idh_pre_dialysis_model.joblib"

THRESHOLD = 0.35

FEATURES = [
    "D_Sexo",
    "D_Edad",
    "D_Dializador",
    "D_Bano",
    "D_Tecnica",
    "D_PesoSeco",
    "D_PesoPre",
    "D_Ganancia",
    "D_TemperaturaBano",
    "D_FlujoSangre",
    "D_FlujoBano",
    "D_ConductividadBano",
    "D_ConductividadBicarbonato"
]


# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# Check that the model expects our 13 features.
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Model feature names or order do not match FEATURES. "
            f"Model expects: {list(model.feature_names_in_)}"
        )


# --------------------------------
# PATIENT INPUT SCHEMA
# --------------------------------

class PatientData(BaseModel):
    D_Sexo: int
    D_Edad: float
    D_Dializador: int
    D_Bano: int
    D_Tecnica: int
    D_PesoSeco: float
    D_PesoPre: float
    D_Ganancia: float
    D_TemperaturaBano: float
    D_FlujoSangre: float
    D_FlujoBano: float
    D_ConductividadBano: float
    D_ConductividadBicarbonato: float


# --------------------------------
# CREATE FASTAPI APPLICATION
# --------------------------------

app = FastAPI(
    title="IDH Care API",
    description="Backend API for IDH Prediction System",
    version="1.0.0"
)


# --------------------------------
# CORS CONFIGURATION
# --------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------
# HOME ENDPOINT
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to IDH Care API",
        "model_loaded": True
    }


# --------------------------------
# HEALTH CHECK
# --------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------
# MODEL INFORMATION
# --------------------------------

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model.steps[-1][1]).__name__,
        "features_required": FEATURES,
        "threshold": THRESHOLD
    }


# --------------------------------
# INDIVIDUAL IDH PREDICTION
# --------------------------------

@app.post("/predict")
def predict_idh(data: PatientData):

    try:
        patient_data = data.dict()

        input_df = pd.DataFrame(
            [patient_data],
            columns=FEATURES
        )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        prediction = int(probability >= THRESHOLD)

        return {
            "prediction": prediction,
            "idh_probability": round(probability, 4),
            "threshold": THRESHOLD,
            "result": (
                "IDH Present"
                if prediction == 1
                else "IDH Absent"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------
# BATCH IDH PREDICTION
# --------------------------------

@app.post("/predict-batch")
def predict_batch(data: list[PatientData]):

    try:
        if not data:
            raise HTTPException(
                status_code=400,
                detail="No patient records provided."
            )

        input_df = pd.DataFrame(
            [patient.dict() for patient in data],
            columns=FEATURES
        )

        probabilities = model.predict_proba(input_df)[:, 1]

        results = []

        for probability in probabilities:

            prediction = int(probability >= THRESHOLD)

            results.append({
                "prediction": prediction,
                "idh_probability": round(float(probability), 4),
                "result": (
                    "IDH Present"
                    if prediction == 1
                    else "IDH Absent"
                )
            })

        return {
            "total_records": len(results),
            "results": results
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
