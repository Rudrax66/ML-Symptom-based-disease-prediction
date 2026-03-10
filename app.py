from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# ─── Load Models ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(f'{BASE}/models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open(f'{BASE}/models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)
with open(f'{BASE}/models/symptoms.json') as f:
    ALL_SYMPTOMS = json.load(f)
with open(f'{BASE}/models/disease_info.json') as f:
    DISEASE_INFO = json.load(f)

@app.route('/')
def index():
    return render_template('index.html', symptoms=ALL_SYMPTOMS)

@app.route('/api/symptoms')
def get_symptoms():
    return jsonify(ALL_SYMPTOMS)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    selected_symptoms = data.get('symptoms', [])
    model_choice = data.get('model', 'rf')

    if not selected_symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Build feature vector (DataFrame to match training feature names)
    import pandas as pd
    features = {s: (1 if s in selected_symptoms else 0) for s in ALL_SYMPTOMS}
    X = pd.DataFrame([features])

    # Select model
    model = rf_model if model_choice == 'rf' else gb_model

    # Predict with probabilities
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    # Top 3 predictions
    top_indices = np.argsort(proba)[::-1][:3]
    predictions = []
    for i in top_indices:
        disease = classes[i]
        conf = round(float(proba[i]) * 100, 1)
        if conf < 0.5:
            continue
        info = DISEASE_INFO.get(disease, {
            "description": "Information not available.",
            "precautions": [],
            "medications": [],
            "diet": [],
            "workout": []
        })
        predictions.append({
            "disease": disease,
            "confidence": conf,
            "description": info.get("description", ""),
            "precautions": info.get("precautions", []),
            "medications": info.get("medications", []),
            "diet": info.get("diet", []),
            "workout": info.get("workout", [])
        })

    return jsonify({
        "predictions": predictions,
        "symptoms_used": selected_symptoms,
        "model": "Random Forest" if model_choice == 'rf' else "Gradient Boosting"
    })

@app.route('/api/disease/<name>')
def disease_info(name):
    info = DISEASE_INFO.get(name)
    if not info:
        return jsonify({'error': 'Disease not found'}), 404
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
