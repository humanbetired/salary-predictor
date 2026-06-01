from fastapi import FastAPI, HTTPException
from app.schemas import JobFeatures, PredictionResponse
from app.model import predict

app = FastAPI(
    title="ML Salary Predictor API",
    description="Prediksi gaji ML Engineer berdasarkan job features",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "salary-predictor-v1"}


@app.post("/predict", response_model=PredictionResponse)
def predict_salary(features: JobFeatures):
    try:
        salary = predict(features.model_dump())
        return PredictionResponse(
            predicted_salary_usd=salary,
            experience_level=features.experience_level,
            company_location=features.company_location,
            model_version="salary-predictor-v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "ML Salary Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }