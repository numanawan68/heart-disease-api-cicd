from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from schema.prediction import HeartDiseaseInput

import pickle
import numpy as np
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = FastAPI(title="Heart Disease Prediction API")


# Load model and scaler
model = pickle.load(open("model/rf_classifier.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


# SHAP explainer
explainer = shap.TreeExplainer(model)


# Feature names
feature_names = [
    "Gender",
    "Age",
    "Current Smoker",
    "Cigarettes Per Day",
    "BP Medication",
    "Stroke History",
    "Hypertension",
    "Diabetes",
    "Cholesterol",
    "Systolic BP",
    "Diastolic BP",
    "BMI",
    "Heart Rate",
    "Glucose"
]


# HTML templates
templates = Jinja2Templates(directory="templates")


# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index2.html",
        context={"request": request}
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(
    male: str = Form(...),
    age: float = Form(...),
    currentSmoker: str = Form(...),
    cigsPerDay: float = Form(...),
    BPMeds: str = Form(...),
    prevalentStroke: str = Form(...),
    prevalentHyp: str = Form(...),
    diabetes: str = Form(...),
    totChol: float = Form(...),
    sysBP: float = Form(...),
    diaBP: float = Form(...),
    BMI: float = Form(...),
    heartRate: float = Form(...),
    glucose: float = Form(...)
):

    try:

        # Validate input using Pydantic schema
        data = HeartDiseaseInput(
            male=male,
            age=age,
            currentSmoker=currentSmoker,
            cigsPerDay=cigsPerDay,
            BPMeds=BPMeds,
            prevalentStroke=prevalentStroke,
            prevalentHyp=prevalentHyp,
            diabetes=diabetes,
            totChol=totChol,
            sysBP=sysBP,
            diaBP=diaBP,
            BMI=BMI,
            heartRate=heartRate,
            glucose=glucose
        )

        # Convert categorical values to numbers
        male_value = 1 if data.male.lower() == "male" else 0
        smoker_value = 1 if data.currentSmoker.lower() == "yes" else 0
        bp_meds_value = 1 if data.BPMeds.lower() == "yes" else 0
        stroke_value = 1 if data.prevalentStroke.lower() == "yes" else 0
        hypertension_value = 1 if data.prevalentHyp.lower() == "yes" else 0
        diabetes_value = 1 if data.diabetes.lower() == "yes" else 0

        # Create feature array
        features = np.array([[
            male_value,
            data.age,
            smoker_value,
            data.cigsPerDay,
            bp_meds_value,
            stroke_value,
            hypertension_value,
            diabetes_value,
            data.totChol,
            data.sysBP,
            data.diaBP,
            data.BMI,
            data.heartRate,
            data.glucose
        ]])

        # Scale input
        scaled = scaler.transform(features)

        # Prediction
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        result = (
            "⚠️ Heart Disease Detected"
            if pred == 1
            else "✅ No Heart Disease"
        )

        # SHAP values
        shap_values = explainer.shap_values(scaled)

        values = shap_values[:, :, 1][0]

        feature_impacts = {}

        for i, feature in enumerate(feature_names):
            feature_impacts[feature] = values[i]

        top_features = sorted(
            feature_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation_text = []

        for feature, impact in top_features:

            effect = (
                "increased"
                if impact > 0
                else "decreased"
            )

            explanation_text.append(
                f"{feature} {effect} your heart disease risk"
            )

        # SHAP plot
        plt.figure(figsize=(8, 4))

        shap.plots.waterfall(
            shap.Explanation(
                values=values,
                base_values=explainer.expected_value[1],
                data=scaled[0],
                feature_names=feature_names
            ),
            show=False
        )

        plt.tight_layout()

        plt.savefig("static/shap_plot.png")

        plt.close()

        return {
            "prediction": (
                f"{result} "
                f"(Risk: {prob * 100:.1f}%)"
            ),
            "explanation": explanation_text,
            "shap_plot": "/static/shap_plot.png"
        }

    except Exception as e:

        return {
            "error": str(e)
        }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
