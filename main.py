
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Locate the saved machine learning model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "idh_logistic_model.joblib"

# Load the trained model pipeline
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Place idh_logistic_model.joblib beside main.py."
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
