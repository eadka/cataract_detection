from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import onnxruntime as ort
from PIL import Image
import numpy as np
import numpy as np
import io


app = FastAPI(title="Cataract Detection API")

# Load ONNX model
BASE_DIR = Path(__file__).resolve().parent  # project root
MODEL_PATH = BASE_DIR / "model" / "cataract_mobilenet_v2_fixed.onnx"
session = ort.InferenceSession(str(MODEL_PATH))

# Model input/output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Normalization constants (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224

# Class labels
# CLASS_LABELS = ["Normal", "Cataract"]
IDX_TO_CLASS = {
    0: "Cataract",
    1: "Normal"
}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Load image bytes, resize, normalize, and convert to NCHW tensor"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)    # Add batch dim
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        outputs = session.run([output_name], {input_name: input_tensor})
        probs = outputs[0][0]  # shape: [num_classes]      
        
        # Convert logits to probabilities using NumPy softmax
        exp_probs = np.exp(probs - np.max(probs))  # subtract max for numerical stability
        probs_softmax = exp_probs / exp_probs.sum()

        class_idx = int(np.argmax(probs_softmax))
        label = IDX_TO_CLASS[class_idx]
        confidence = float(probs_softmax[class_idx])

        return JSONResponse(
            content={
                "prediction": label,
                "confidence": confidence
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}
