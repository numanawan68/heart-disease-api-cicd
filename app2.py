from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import shap
import matplotlib

matplotlib.use('Agg')  # important for flask
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# Feature names
feature_names = [
    'Gender',
    'Age',
    'Current Smoker',
    'Cigarettes Per Day',
    'BP Medication',
    'Stroke History',
    'Hypertension',
    'Diabetes',
    'Cholesterol',
    'Systolic BP',
    'Diastolic BP',
    'BMI',
    'Heart Rate',
    'Glucose'
]


@app.route("/")
def index():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.form

        male = 1 if data.get('male').lower() == 'male' else 0
        age = float(data.get('age'))
        currentSmoker = 1 if data.get('currentSmoker').lower() == 'yes' else 0
        cigsPerDay = float(data.get('cigsPerDay'))
        BPMeds = 1 if data.get('BPMeds').lower() == 'yes' else 0
        prevalentStroke = 1 if data.get('prevalentStroke').lower() == 'yes' else 0
        prevalentHyp = 1 if data.get('prevalentHyp').lower() == 'yes' else 0
        diabetes = 1 if data.get('diabetes').lower() == 'yes' else 0
        totChol = float(data.get('totChol'))
        sysBP = float(data.get('sysBP'))
        diaBP = float(data.get('diaBP'))
        BMI = float(data.get('BMI'))
        heartRate = float(data.get('heartRate'))
        glucose = float(data.get('glucose'))

        # Features array
        features = np.array([[
            male, age, currentSmoker, cigsPerDay,
            BPMeds, prevalentStroke, prevalentHyp,
            diabetes, totChol, sysBP,
            diaBP, BMI, heartRate, glucose
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

        # -----------------------------
        # SHAP EXPLANATION
        # -----------------------------
        shap_values = explainer.shap_values(scaled)

        feature_impacts = {}

        # Binary classification
        values = shap_values[:, :, 1][0]

        for i, feature in enumerate(feature_names):
            feature_impacts[feature] = values[i]

        top_features = sorted(
            feature_impacts.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation_text = []

        for feature, impact in top_features:
            effect = "increased" if impact > 0 else "decreased"

            explanation_text.append(
                f"{feature} {effect} your heart diseas risk"
            )

        # -----------------------------
        # SHAP BAR PLOT
        # -----------------------------
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
        plt.savefig('static/shap_plot.png')
        plt.close()

        return jsonify({
            'prediction':
                f"{result} "
                f"(Risk: {prob * 100:.1f}%)",

            'explanation':
                explanation_text,

            'shap_plot':
                '/static/shap_plot.png'
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)