import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import io

# --------------------
# Config
# --------------------
# Load ONNX model
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = BASE_DIR / "model" / "cataract_mobilenet_v2_fixed.onnx"

IDX_TO_CLASS = {
    0: "Cataract",
    1: "Normal"
}

# --------------------
# Load model once
# --------------------
@st.cache_resource
def load_model():
    return ort.InferenceSession(MODEL_PATH)

session = load_model()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --------------------
# Preprocessing
# --------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)   # add batch dim

    return img_array

# --------------------
# UI
# --------------------
st.title("Cataract Detection 👁️")
st.write("Upload a retinal image to detect cataract.")


uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)
# st.write("Click to upload an image (drag & drop may be unavailable in some environments)")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    image_bytes = uploaded_file.getvalue()
    input_tensor = preprocess_image(image_bytes)

    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0][0]

    # Softmax (numpy)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    class_idx = int(np.argmax(probs))
    label = IDX_TO_CLASS[class_idx]
    confidence = float(probs[class_idx]) * 100

    st.markdown("### Prediction")
    st.success(f"**{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
