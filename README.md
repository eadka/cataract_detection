# 👁️ Cataract Detection
**End-to-end ML system for cataract detection with deep learning, Docker, and Kubernetes**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML Zoomcamp](https://img.shields.io/badge/ML%20Zoomcamp-DataTalksClub-orange)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind-blueviolet)
![Cloud](https://img.shields.io/badge/Cloud-Fly.io-purple)

---
## Problem description


---

## Repository Structure

```text
cataract-detection/
│
├── data/
│   └── sample_images/          # Sample eye images for testing/demo
│
├── notebooks/
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Model evaluation and metrics
│
├── model/
│   └── model.h5                # Trained CNN model 
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


--- 

## EDA


---

## Model Training

---

## Exporting notebook to script

---

## Reproducibility

---

## Model Deployment

---

## Dependency and enviroment management


---

## Containerization


--- 

## Cloud deployment


