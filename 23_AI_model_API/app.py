# ============================================================
# REST API FOR BREAST CANCER PREDICTION USING FASTAPI
# ============================================================

import time
import joblib
import logging
import numpy as np

from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    filename="logs/api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================

try:
    small_model = joblib.load("models/small_model.pkl")
    large_model = joblib.load("models/large_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    metadata = joblib.load("models/model_metadata.pkl")
    label_mapping = joblib.load("models/label_mapping.pkl")

    logging.info("All model files loaded successfully.")

except Exception as error:
    logging.error(f"Model loading failed: {error}")
    raise RuntimeError("Model files could not be loaded. Please check the models folder.")


# ============================================================
# FASTAPI APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Breast Cancer Prediction REST API",
    description="A professional FastAPI REST API for serving small and large machine learning models.",
    version="1.0.0"
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):
    """
    Request schema for single prediction.

    model_type:
        small = Logistic Regression
        large = Random Forest

    features:
        Must contain exactly 30 numerical values from the Breast Cancer dataset.
    """

    model_type: Literal["small", "large"] = Field(
        default="small",
        description="Choose 'small' for low memory or 'large' for better performance."
    )

    features: List[float] = Field(
        ...,
        description="List of 30 numerical input features."
    )

    @field_validator("features")
    @classmethod
    def validate_features_length(cls, value):
        if len(value) != 30:
            raise ValueError("Exactly 30 feature values are required.")
        return value


class BatchRecord(BaseModel):
    """
    Schema for one record inside batch prediction.
    """

    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features_length(cls, value):
        if len(value) != 30:
            raise ValueError("Exactly 30 feature values are required for each record.")
        return value


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch prediction.
    """

    model_type: Literal["small", "large"] = Field(default="small")
    records: List[BatchRecord]


# ============================================================
# HELPER FUNCTION: SELECT MODEL
# ============================================================

def select_model(model_type: str):
    """
    Selects the requested model.

    small:
        Logistic Regression model for low-memory deployment.

    large:
        Random Forest model for stronger prediction performance.
    """

    if model_type == "small":
        return small_model

    if model_type == "large":
        return large_model

    raise HTTPException(
        status_code=400,
        detail="Invalid model_type. Use either 'small' or 'large'."
    )


# ============================================================
# HELPER FUNCTION: MAKE PREDICTION
# ============================================================

def make_prediction(features: List[float], model_type: str):
    """
    Scales input features, performs prediction, calculates confidence,
    and returns a structured prediction response.
    """

    try:
        model = select_model(model_type)

        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        prediction = int(model.predict(scaled_input)[0])

        confidence = None
        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            confidence = round(float(np.max(probabilities)), 4)
            probabilities = [round(float(prob), 4) for prob in probabilities]

        predicted_label = label_mapping[prediction]

        return {
            "model_used": model_type,
            "prediction": prediction,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": probabilities
        }

    except Exception as error:
        logging.error(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the input values."
        )


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    """
    Root endpoint to confirm that the API is running.
    """

    return {
        "message": "Breast Cancer Prediction API is running successfully.",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model-info"
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health_check():
    """
    Checks API and model loading status.
    """

    return {
        "status": "healthy",
        "small_model_loaded": small_model is not None,
        "large_model_loaded": large_model is not None,
        "scaler_loaded": scaler is not None
    }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():
    """
    Returns model metadata saved during notebook training.
    """

    return metadata


# ============================================================
# SINGLE PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Accepts one input record and returns prediction result.
    """

    start_time = time.time()

    result = make_prediction(
        features=request.features,
        model_type=request.model_type
    )

    response_time = round(time.time() - start_time, 4)

    logging.info(
        f"Single prediction completed | Model: {request.model_type} | Time: {response_time}s"
    )

    return {
        "success": True,
        "input_features": request.features,
        "result": result,
        "response_time_seconds": response_time
    }


# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    """
    Accepts multiple records and returns predictions for all records.
    """

    start_time = time.time()

    if len(request.records) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch records cannot be empty."
        )

    predictions = []

    for index, record in enumerate(request.records):
        result = make_prediction(
            features=record.features,
            model_type=request.model_type
        )

        predictions.append({
            "record_number": index + 1,
            "input_features": record.features,
            "result": result
        })

    response_time = round(time.time() - start_time, 4)

    logging.info(
        f"Batch prediction completed | Records: {len(request.records)} | Model: {request.model_type}"
    )

    return {
        "success": True,
        "model_used": request.model_type,
        "total_records": len(request.records),
        "predictions": predictions,
        "response_time_seconds": response_time
    }