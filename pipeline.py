
from prefect import flow, task
from services.data_pipeline import load_data, preprocess_data
from services.ml_pipeline import train_models

@task
def data_task():
    df = load_data()
    df = preprocess_data(df)
    return df

@task
def ml_task():
    return train_models()

@flow(name="war-data-pipeline")
def pipeline():
    df = data_task()
    metrics = ml_task()
    return metrics

if __name__ == "__main__":
    pipeline()
