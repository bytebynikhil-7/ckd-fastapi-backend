from fastapi import FastAPI
from app.schemas import PredictionRequest
from app.predictor import load_models, predict

app = FastAPI()

@app.on_event("startup")
def startup():
    load_models()

@app.get("/")
def root():
    return {"message": "CKD API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_ckd(request: PredictionRequest):

    features = [
        request.sg,
        request.htn,
        request.hemo,
        request.dm,
        request.al,
        request.appet,
        request.rc,
        request.pc
    ]

    prediction, confidence = predict(
        request.model,
        features
    )

    return {
        "prediction": int(prediction),
        "confidence": confidence,
        "model": request.model
    }
