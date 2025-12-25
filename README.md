#  Cataract Detection ML System
**End-to-end ML system for cataract detection with deep learning, Docker, and Kubernetes**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML Zoomcamp](https://img.shields.io/badge/ML%20Zoomcamp-DataTalksClub-orange)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-kind-blueviolet)
![Cloud](https://img.shields.io/badge/Cloud-Fly.io-purple)



<p align="center">
  <img src="images/icons8-ophthalmology-30.png" alt="Eye image" width="120"/>
</p>

<h1 align="center">Cataract Detection ML System</h1>

<p align="center">
  End-to-end ML system for cataract detection with deep learning, Docker, and Kubernetes
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue"/>
  <img src="https://img.shields.io/badge/ML%20Zoomcamp-DataTalksClub-orange"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-blue"/>
  <img src="https://img.shields.io/badge/Kubernetes-kind-blueviolet"/>
  <img src="https://img.shields.io/badge/Cloud-Fly.io-purple"/>
</p>


## Repo Structure

cataract-detection/
│
├── data/
│   └── sample_images/
│
├── model/
│   ├── train.py
│   ├── evaluate.py
│   └── model.h5
│
├── app/
│   ├── app.py        # FastAPI
│   └── predict.py
│
├── streamlit_app/
│   └── ui.py
│
├── docker/
│   └── Dockerfile
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── requirements.txt
├── README.md
└── Makefile
