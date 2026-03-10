# 🏥 MediPredict AI — Disease Prediction System

A Flask + Machine Learning web app that predicts diseases from symptoms and provides comprehensive health guidance.

## Features
- **41 diseases**, **138 symptoms** — full symptom-based prediction
- **Dual ML models**: Random Forest & Gradient Boosting (~99.9% accuracy on test set)
- **Top-3 ranked predictions** with confidence scores
- **4 health categories** per disease: Precautions · Medications · Diet · Workout
- Live symptom search & filtering
- Beautiful dark medical UI with animations

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (one-time)
python train_model.py

# 3. Run Flask app
python app.py

# 4. Open browser
http://localhost:5000
```

## Project Structure
```
disease_predictor/
├── app.py                  # Flask API
├── train_model.py          # ML training script
├── requirements.txt
├── models/
│   ├── rf_model.pkl        # Random Forest model
│   ├── gb_model.pkl        # Gradient Boosting model
│   ├── symptoms.json       # Symptom list
│   └── disease_info.json   # Disease knowledge base
└── templates/
    └── index.html          # Frontend UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/predict` | POST | Predict disease from symptoms |
| `/api/symptoms` | GET | Get all symptoms |
| `/api/disease/<name>` | GET | Get disease info |

### Predict Request
```json
POST /api/predict
{
  "symptoms": ["fever", "headache", "vomiting"],
  "model": "rf"   // "rf" or "gb"
}
```

## Diseases Covered
Fungal infection, Allergy, GERD, Diabetes, Malaria, Dengue, Typhoid, Pneumonia, Hepatitis (A/B/C/D/E), Tuberculosis, Hypertension, Migraine, Asthma, Heart attack, Arthritis, Psoriasis, Acne, UTI, and 20+ more.

---
> ⚠️ **Disclaimer**: For educational purposes only. Not a substitute for professional medical advice.
