# Breast Cancer Prediction REST API using FastAPI

A professional REST API project that serves trained Machine Learning models using FastAPI. The API predicts whether a breast cancer tumor is **Malignant** or **Benign** using the Scikit-Learn Breast Cancer dataset.

This project supports both a lightweight model and a larger performance-focused model, making it suitable for systems with different memory and deployment constraints.

---

# Project Objective

The objective of this project is to build a REST API that can serve trained AI/ML models and allow external applications to send input data through HTTP requests and receive predictions in JSON format.

The API is built using FastAPI and includes model loading, input validation, error handling, logging, single prediction, batch prediction, Docker support, and deployment-ready structure.

---

# Features

- REST API using FastAPI
- Small Model Support (Low Memory)
- Large Model Support (High Performance)
- Breast Cancer Classification
- Pydantic Input Validation
- Single Prediction Endpoint
- Batch Prediction Endpoint
- Health Check Endpoint
- Model Metadata Endpoint
- Confidence Score Output
- Structured JSON Response
- Logging System
- Exception Handling
- Docker Support
- Swagger Documentation
- Deployment Ready

---

# Machine Learning Models

## Small Model

**Logistic Regression**

A lightweight model designed for low-memory environments and faster inference.

## Large Model

**Random Forest Classifier**

A more powerful model designed for higher predictive performance.

---

# Project Structure

```text
23_AI_Model_REST_API/
│
├── logs/
│   └── api_logs.log
│
├── models/
│   ├── small_model.pkl
│   ├── large_model.pkl
│   ├── scaler.pkl
│   ├── model_metadata.pkl
│   └── label_mapping.pkl
│
├── test_requests/
│   ├── sample_request.json
│   └── sample_batch_request.json
│
├── ai_model_api.ipynb
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Tech Stack

- Python
- FastAPI
- Scikit-Learn
- NumPy
- Pandas
- Joblib
- Pydantic
- Uvicorn
- Docker

---

# API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Home Endpoint |
| `/health` | GET | API Health Check |
| `/model-info` | GET | Model Metadata |
| `/predict` | POST | Single Prediction |
| `/predict-batch` | POST | Batch Prediction |

---

# Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc Documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# Example Single Prediction Request

### Endpoint

```text
POST /predict
```

### Request Body

```json
{
  "model_type": "small",
  "features": [
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189
  ]
}
```

### Example Response

```json
{
  "success": true,
  "result": {
    "model_used": "small",
    "prediction": 0,
    "predicted_label": "malignant",
    "confidence": 0.9823
  },
  "response_time_seconds": 0.0021
}
```

---

# Example Batch Prediction Request

### Endpoint

```text
POST /predict-batch
```

### Request Body

```json
{
  "model_type": "large",
  "records": [
    {
      "features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
    },
    {
      "features": [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
    }
  ]
}
```

---

# Testing with Postman

1. Open Postman
2. Create a new POST request
3. Enter:

```text
http://127.0.0.1:8000/predict
```

4. Select **Body**
5. Select **Raw**
6. Choose **JSON**
7. Paste the sample request
8. Click **Send**

---

# Docker Support

## Build Docker Image

```bash
docker build -t breast-cancer-api .
```

## Run Docker Container

```bash
docker run -p 8000:8000 breast-cancer-api
```

Access the API:

```text
http://127.0.0.1:8000/docs
```

---

# Deployment

This API can be deployed on:

- Render
- Railway
- AWS
- Azure
- Google Cloud Platform

### Render Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

# Bonus Features Implemented

- Input Validation using Pydantic
- Health Check Endpoint
- Model Information Endpoint
- Batch Prediction Support
- Confidence Scores
- Logging System
- Error Handling
- Docker Support
- Swagger Documentation
- Small and Large Model Selection
- Structured JSON Responses
- Cloud Deployment Ready

---

# Important Disclaimer

This project uses the Breast Cancer dataset from Scikit-Learn for educational and demonstration purposes only.

The predictions generated by this API should not be considered medical advice and must not be used for real-world medical diagnosis.

---

# Author

**Saif Ullah**

Artificial Intelligence Engineer

---

# Project Status

Completed and Ready for:

- Local Deployment
- API Testing
- Docker Deployment
- Cloud Deployment
- Portfolio Showcase