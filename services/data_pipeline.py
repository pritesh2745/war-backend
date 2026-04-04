
import pandas as pd
import numpy as np

def load_data():
    return pd.read_csv("data/waves.csv")

def preprocess_data(df):
    df = df.fillna(0)
    numeric_cols = ["drones", "missiles", "munitions", "fatalities", "injuries"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df["total_attacks"] = df["drones"] + df["missiles"] + df["munitions"]
    return df

def get_summary(df):
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }

def get_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().to_dict()
