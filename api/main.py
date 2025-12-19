from fastapi import FastAPI
import joblib

class PredictionRequest(BaseModel): # <-- Add this class
    text: str

app = FastAPI()
model, vectorizer = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(request: PredictionRequest):
    vec = vectorizer.transform()
    return {"prediction": model.predict(vec)}
