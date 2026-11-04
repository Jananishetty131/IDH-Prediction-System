
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Locate the saved machine learning model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "idh_pre_dialysis_model.joblib"

# Load the trained model pipeline
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Place idh_pre_dialysis_model.joblib beside main.py."
    )

model = joblib.load(MODEL_PATH)

# Create the FastAPI application
app = FastAPI(
    title="IDH Care API",
    description="Backend API for the IDH prediction project",
    version="1.0.0",
)

# Allow our local React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to IDH Care API",
        "model_loaded": True,
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model.named_steps["classifier"]).__name__,
        "features_required": list(model.feature_names_in_),
    }
    
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


@app.post("/predict")
def predict_idh(data: PatientData):
    try:
        patient_data = data.dict()
        prediction_input = pd.DataFrame([patient_data])

        probability = model.predict_proba(prediction_input)[0][1]

        threshold = 0.65
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "idh_probability": round(float(probability), 4),
            "result": "IDH Present" if prediction == 1 else "IDH Absent"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
