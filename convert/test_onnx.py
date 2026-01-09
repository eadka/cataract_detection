import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model/cataract_mobilenet_v2_fixed.onnx")

dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = session.run(None, {"input": dummy})

print(outputs[0].shape)
