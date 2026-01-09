import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import torch
from src.model import CataractDetectorMobileNet

# Load your trained model
MODEL_URL = (
    "https://github.com/eadka/cataract_detection/releases/download/"
    "v1.0-mobilenet-cataract/mobilenet_v4_06_0.980.pth"
)

# Load model
state_dict = torch.hub.load_state_dict_from_url(
    MODEL_URL,
    map_location="cpu",
    progress=True
)

model = CataractDetectorMobileNet()
model.load_state_dict(state_dict)
model.eval()

# Fixed batch size = 1
dummy_input = torch.randn(1, 3, 224, 224)

# Disable TorchDynamo
torch._dynamo.disable()

# Export to ONNX with all weights embedded, no dynamic axes
torch.onnx.export(
    model,
    dummy_input,
    "model/cataract_mobilenet_v2_fixed.onnx",
    opset_version=18,           # latest recommended
    input_names=["input"],
    output_names=["output"],
    export_params=True,         # embed all weights
    do_constant_folding=True,   # optimize constants
    dynamo=False # force legacy exporter for single-file ONNX
)
print("✅ Exported single-file ONNX model: cataract_mobilenet_v2_fixed.onnx")
