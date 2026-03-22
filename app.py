from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load dataset (only for summary endpoint)
df = pd.read_csv("waves.csv")

# 🔥 DIRECT SCORING FUNCTION (NO ML MODEL)
def calculate_intensity(drones, missiles, munitions, fatalities, injuries):
    score = (
        drones * 1 +
        missiles * 3 +
        munitions * 0.7 +
        fatalities * 10 +
        injuries * 3
    )

    if score < 100:
        return "LOW"
    elif score < 300:
        return "MEDIUM"
    else:
        return "HIGH"

@app.route("/")
def home():
    return "War Intelligence API Running (Final Logic)"

@app.route("/predict")
def predict():
    drones = float(request.args.get("drones", 10))
    missiles = float(request.args.get("missiles", 5))
    munitions = float(request.args.get("munitions", 50))
    fatalities = float(request.args.get("fatalities", 1))
    injuries = float(request.args.get("injuries", 2))

    intensity = calculate_intensity(
        drones, missiles, munitions, fatalities, injuries
    )

    return jsonify({"intensity": intensity})

@app.route("/summary")
def summary():
    return jsonify({
        "records": len(df),
        "note": "Using rule-based scoring system"
    })

if __name__ == "__main__":
    app.run(debug=True)
