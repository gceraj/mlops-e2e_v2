import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/clean.csv")
X, y = df.review_text, df.sentiment

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=300)
model.fit(Xtr, ytr)

preds = model.predict(Xte)
acc = accuracy_score(yte, preds)

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)

joblib.dump((model, vectorizer), "models/model.pkl")
print("Accuracy:", acc)
