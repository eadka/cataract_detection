# 👁️ Cataract Detection
<div align="center">
  <img 
    src="https://collaborativeeye.com/wp-content/uploads/sites/7/2019/02/The-Importance-of-the-Ocular-Surface-in-Cataract-Surgery-Hero-740x366.jpg"
    style="width:75%; object-fit:contain;"
    alt="Cataract illustration"
  />
</div>

**End-to-end ML system for cataract detection with deep learning, Docker, and Kubernetes**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML Zoomcamp](https://img.shields.io/badge/ML%20Zoomcamp-DataTalksClub-orange)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind-blueviolet)
![Cloud](https://img.shields.io/badge/Cloud-Fly.io-purple)

KaggleOpen Streamlit App

This repository was created as part of the DataTalks.Club's Machine Learning Zoomcamp by Alexey Grigorev.

This project has been submitted as the Capstone project for the course.

---
## Overview
A cataract is a common eye condition in which the normally clear lens of the eye becomes cloudy, leading to blurred or impaired vision. The lens plays a critical role in focusing light onto the retina, and any loss of its transparency can significantly affect visual clarity.

Cataracts most commonly develop with aging and are one of the leading causes of visual impairment worldwide. While cataracts can occur at any age, including congenital cases, age-related cataracts account for the majority of diagnoses. When detected early, visual symptoms may be mild; however, advanced cataracts can lead to significant vision loss if left untreated.

## Causes
Cataracts develop when changes occur in the proteins and fibers within the eye’s lens, causing them to clump together and scatter light. These changes are most often associated with aging but may also be influenced by several other factors.

Common causes and risk factors include:

- Aging (the most significant risk factor)
- Prolonged exposure to ultraviolet (UV) radiation
- Diabetes mellitus
- Smoking and excessive alcohol consumption
- Eye trauma or previous eye surgery
- Long-term use of corticosteroid medications
- Genetic predisposition or congenital conditions

These factors can accelerate lens protein degeneration, leading to progressive clouding and reduced visual acuity.

## Clinical Significance

Cataracts are typically diagnosed through a comprehensive eye examination and are often treatable with surgical intervention. Cataract surgery involves replacing the cloudy lens with an artificial intraocular lens (IOL) and has a high success rate.

Early detection plays an important role in preventing severe vision impairment. Automated cataract detection systems using medical imaging and deep learning models, such as convolutional neural networks, have the potential to support clinical screening and improve accessibility to eye care services.

## Repository Structure

```text
cataract-detection/
│
├── src/
│   ├── model.py              # ✅ model architecture (THIS is Step 1)
│   ├── train.py              # training loop
│   ├── evaluate.py           # validation / test evaluation
│   ├── predict.py            # inference script (optional)
│
├── convert/
│   ├── convert_to_onnx.py    # ONNX export script
│
├── data/
│   └── sample_images/          # Sample eye images for testing/demo
│
├── notebooks/
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Model evaluation and metrics
│
├── model/
│   └── mobilenet_v4_06_0.980.pth                # Trained CNN model 
├── app/
│   ├── app.py                  # FastAPI inference service
│   └── predict.py              # Prediction logic and preprocessing
│
├── streamlit_app/
│   └── ui.py                   # Streamlit user interface
│
├── docker/
│   └── Dockerfile              # Dockerfile for inference service
│
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment
│   └── service.yaml            # Kubernetes Service
│
│── images/
│
├── requirements.txt            # Python dependencies
├── Makefile                    # Common project commands
└── README.md                   # Project documentation


/kaggle/working/split_data/
  ├── train/
  │   ├── cataract/
  │   └── normal/
  └── val/
      ├── cataract/
      └── normal/
```
---
## Dataset
In this project, the following Kaggle dataset has been used: [Cataract Dataset](https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset)

![Cataract Dataset](images/cataract_dataset.png)

The dataset can be downloaded with the following code:
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nandanp6/cataract-image-dataset")

print("Path to dataset files:", path)
```
## About the dataset

The images in this dataset are cataract eye images dataset from camera captures which have been scrapped from the web. 

The dataset has 612 images split as `cataract` and `normal` under train and test folders. 

The train dataset has 491 images: 245 as cataract and 246 as normal. 

The test has 121 images: 61 as cataract and 60 as normal.

## EDA


---

## Model Training


## Trained Model

The final trained model is stored as a versioned artifact using GitHub Releases, separate from the training code, to ensure reproducibility and clean version control and can be found below:

- **Model**: MobileNetV2 Cataract Detector
- **Version**: v1.0
- **Test Accuracy**: 98.35%

📦 Download Mobilenet Pytorch Model:
```
wget https://github.com/eadka/cataract_detection/releases/download/v1.0-mobilenet-cataract/mobilenet_v4_06_0.980.pth
```

Next, the model trained in PyTorch is exported to ONNX, and served via FastAPI to create a lightweight, portable, and production-ready inference service. The ONNX model can be downloaded from the below link:

📦 Download ONNX Model:
```
wget https://github.com/eadka/cataract_detection/releases/download/v1.0.1-mobilenet-cataract-onnx/cataract_mobilenet_v2_fixed.onnx
```

---

## Exporting notebook to script

---

## Reproducibility

---

## Model Deployment

### 🧠 FastAPI + ONNX
The trained cataract detection model is exported to ONNX format and served via a FastAPI application. This allows fast, portable, and framework-agnostic inference.

Key Components:
- ONNX Runtime
  - Used for efficient model inference (onnxruntime)
  - Enables lightweight deployment without PyTorch/TensorFlow
- FastAPI
  - Exposes a /predict endpoint
  - Accepts image uploads via multipart form data
  - Returns prediction label and confidence score as JSON
- Uvicorn
  - ASGI server for local and containerized deployment

To run it, follow the below steps
```
cd serve

uv run uvicorn app:app --reload
```

#### API Endpoint: Swagger UI
POST /predict:
👉 http://127.0.0.1:8000/docs

Upload an image and click on Execute to check the result.

![Fastapi Endpoint](images/fastapi.png)

The result should appear as:
```
{
  "prediction": "Normal",
  "confidence": 0.9998186230659485
}
```

#### API Endpoint: Python script using requests

The python script [fastapi_test.py](https://github.com/eadka/cataract_detection/tree/main/serve/fastapi_test.py) uses the requests library to test the FastAPI /predict endpoint by sending an image file and printing the model’s response.

Two sample images are provided in the [images folder](https://github.com/eadka/cataract_detection/tree/main/images) for testing:
  - `test_image_cataract.png`
  - `test_image_normal.png`

You can modify the image path in `fastapi_test.py` to test different images.

##### Run the test

Make sure the FastAPI server is running, then open a new terminal and run:

```
cd serve

uv run python ./fastapi_test.py

```
##### Example output
```
{
  'prediction': 'Cataract', 
  'confidence': 0.9998255372047424
}
```
This confirms that the API endpoint is working correctly and returning model predictions.

###  Streamlit Web App

A Streamlit application is used to provide a simple and interactive web interface for running cataract detection on uploaded eye images. The app allows users to upload an image and view the model’s prediction and confidence score in real time.

The Streamlit app reuses the same preprocessing logic and ONNX model as the FastAPI service, ensuring consistent predictions across both interfaces.

##### Key Components
- Streamlit
  - Provides a lightweight web UI for model inference
  - Handles image upload and display
  - Shows prediction label and confidence score
- ONNX Runtime
  - Used to load and run the exported ONNX model
  - Ensures fast inference without requiring PyTorch at runtime
- NumPy & Pillow
  - Used for image preprocessing and tensor preparation

##### Run the Streamlit App
From the project root, run:
```
cd streamlit_app

uv run streamlit run app.py
```
Once running, open the app in your browser at:
👉 http://localhost:8501

##### Using the App

Upload an eye image (.png, .jpg, or .jpeg) using the file uploader.
- The uploaded image is displayed on the page.
- The model predicts whether the image shows Cataract or Normal.
- The prediction label and confidence score are shown below the image.

Note: In some environments (e.g., GitHub Codespaces), drag-and-drop upload may be disabled due to browser iframe restrictions. Click-to-upload works as expected, and drag-and-drop functions normally in local and deployed environments.

##### Example Output
```
Prediction: Cataract
Confidence: 0.9998
```

Screenshot of the Streamlit app:
![Streamlit app](images/streamlit.png)

This Streamlit interface provides a user-friendly way to interact with the model and complements the FastAPI service by offering a visual, non-technical entry point for inference.

## Dependency and enviroment management


---

## Containerization


--- 

## Cloud deployment


