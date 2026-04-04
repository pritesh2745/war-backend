
from fastapi import FastAPI, Query
import joblib

from services.data_pipeline import load_data, preprocess_data, get_summary, get_correlation
from services.ml_pipeline import train_models

app = FastAPI(title="War Intelligence API 🚀")

# Load model globally
model = None

def load_model():
    global model
    try:
        model = joblib.load("models/model.pkl")
    except:
        model = None

load_model()

@app.get("/")
def home():
    return {"message": "War Intelligence API Running 🚀"}

# TRAIN MODEL
@app.get("/train")
def train():
    metrics = train_models()
    load_model()
    return {
        "message": "Model trained successfully",
        "metrics": metrics
    }

# PREDICT USING ML
@app.get("/predict")
def predict(
    drones: float,
    missiles: float,
    munitions: float,
    fatalities: float,
    injuries: float
):
    if model is None:
        return {"error": "Model not trained yet"}

    input_data = [[drones, missiles, munitions, fatalities, injuries]]
    prediction = model.predict(input_data)[0]

    label_map = {
        0: "LOW",
        1: "MEDIUM",
        2: "HIGH"
    }

    return {"intensity": label_map[prediction]}

# DATA SUMMARY
@app.get("/summary")
def summary():
    df = load_data()
    df = preprocess_data(df)
    return get_summary(df)

# CORRELATION
@app.get("/correlation")
def correlation():
    df = load_data()
    df = preprocess_data(df)
    return get_correlation(df)

# EVENTS
@app.get("/events")
def events():
    df = load_data()
    return df.head(50).to_dict(orient="records")

# METRICS
@app.get("/metrics")
def metrics():
    return {"note": "Use /train to generate metrics via MLflow"}

# PIPELINE STATUS
@app.get("/pipeline-status")
def pipeline_status():
    return {
        "status": "Active",
        "schedule": "Every 3 minutes",
        "components": ["Data Pipeline", "ML Training", "MLflow Logging"]
    }
