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

📦 Download:
```
wget https://github.com/eadka/cataract_detection/releases/download/v1.0-mobilenet-cataract/mobilenet_v4_06_0.980.pth
```

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


