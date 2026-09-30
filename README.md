# Heart-Disease-Prediction-System-Python-machine-learning
# Heart Disease Prediction API

A production-ready **Heart Disease Prediction application** built with Machine Learning, FastAPI, Docker, GitHub Actions, Docker Hub, and Render.

## 🚀 Live Demo

**Live App:** https://heart-disease-ml-api-y7v4.onrender.com/

## 🛠️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Pydantic
* SHAP
* HTML/CSS/JavaScript
* Docker
* GitHub Actions
* Docker Hub
* Render

## ✨ Features

* Heart disease prediction using Random Forest
* Probability-based risk prediction
* SHAP explainability
* FastAPI backend
* Web-based frontend
* Automated tests with Pytest
* Dockerized application
* CI/CD with GitHub Actions
* Automatic deployment to Render

## 🔄 CI/CD Pipeline

```text
GitHub
   ↓
GitHub Actions
   ↓
Run Tests
   ↓
Build Docker Image
   ↓
Push to Docker Hub
   ↓
Deploy to Render
```

## ▶️ Run Locally

```bash
git clone https://github.com/numanawan68/heart-disease-api-cicd.git
cd heart-disease-api-cicd

pip install -r requirements.txt

uvicorn app:app --reload
```

Open:

```text
http://localhost:8000
```

## 📌 API

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict
```

## ⚠️ Disclaimer

This project is for educational and demonstration purposes only and is **not a medical diagnostic tool**.
