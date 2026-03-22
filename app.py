from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

df = pd.read_csv("waves.csv")

df = df[[
    'drones_used',
    'ballistic_missiles_used',
    'estimated_munitions_count',
    'fatalities',
    'injuries'
]].fillna(0)

def classify(row):
    score = row['estimated_munitions_count'] + row['fatalities'] * 2
    if score < 50:
        return 0
    elif score < 150:
        return 1
    else:
        return 2

df['intensity'] = df.apply(classify, axis=1)

X = df.drop('intensity', axis=1)
y = df['intensity']

model = RandomForestClassifier()
model.fit(X, y)

@app.route("/")
def home():
    return "War Intelligence API Running"

@app.route("/predict")
def predict():
    drones = float(request.args.get("drones", 10))
    missiles = float(request.args.get("missiles", 5))
    munitions = float(request.args.get("munitions", 50))
    fatalities = float(request.args.get("fatalities", 1))
    injuries = float(request.args.get("injuries", 2))

    sample = [[drones, missiles, munitions, fatalities, injuries]]
    pred = model.predict(sample)[0]

    labels = ["LOW", "MEDIUM", "HIGH"]

    return jsonify({"intensity": labels[pred]})

@app.route("/summary")
def summary():
    return jsonify({
        "records": len(df),
        "features": list(X.columns)
    })

if __name__ == "__main__":
    app.run(debug=True)
