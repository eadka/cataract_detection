import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
IMAGE_PATH = BASE_DIR / "images" / "test_image_cataract.png"
# IMAGE_PATH = BASE_DIR / "images" / "test_image_normal.png"

url = "http://127.0.0.1:8000/predict"
files = {"file": open(IMAGE_PATH, "rb")}

response = requests.post(url, files=files)
print(response.json())