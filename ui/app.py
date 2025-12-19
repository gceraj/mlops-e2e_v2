import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Sentiment Analysis")

text = st.text_area("Enter feedback")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(API_URL, params={"text": text}, timeout=3)
            response.raise_for_status()
            st.success(response.json())
        except requests.exceptions.ConnectionError:
            st.error("🚨 API is not running. Start FastAPI first.")
        except Exception as e:
            st.error(f"Error: {e}")
