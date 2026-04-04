
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from services.data_pipeline import load_data, preprocess_data

MODEL_PATH = "models/model.pkl"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("war-intelligence")

def train_models():
    os.makedirs("models", exist_ok=True)

    df = load_data()
    df = preprocess_data(df)

    def label(row):
        score = (
            row["drones"] * 1 +
            row["missiles"] * 3 +
            row["munitions"] * 0.7 +
            row["fatalities"] * 10 +
            row["injuries"] * 3
        )
        if score < 100:
            return 0
        elif score < 300:
            return 1
        else:
            return 2

    df["target"] = df.apply(label, axis=1)

    X = df[["drones", "missiles", "munitions", "fatalities", "injuries"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    start_time = time.time()

    with mlflow.start_run():
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_param("model", "RandomForest")
        mlflow.sklearn.log_model(rf, "model")

        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

    joblib.dump(rf, MODEL_PATH)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_time": training_time
    }
