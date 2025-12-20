from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class PredictionRequest(BaseModel):
    text: str

app = FastAPI()
model, vectorizer = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(request: PredictionRequest):
    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]
    return {"prediction": prediction}
