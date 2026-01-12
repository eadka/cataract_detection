import streamlit as st
import requests
from PIL import Image
import io
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000/predict")

st.set_page_config(page_title="Cataract Detection", layout="centered")

st.title("👁️ Cataract Detection")
st.write("Upload an eye image to check for cataract using a trained ML model.")

uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Run Prediction"):
        with st.spinner("Sending image to model..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}**")
                st.write(f"Confidence: **{result['confidence']:.4f}**")
            else:
                st.error("Error calling prediction API")
