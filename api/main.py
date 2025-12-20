from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

model, vectorizer = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(request: PredictionRequest):
    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]
    return {"prediction": prediction}


