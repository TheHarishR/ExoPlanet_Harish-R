import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("models\\scaler (2).pkl")

def preprocess_input(data, model_features):
    df = pd.DataFrame([data])

    # Safety
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Ensure feature order
    df = df[model_features]

    # Scaling
    df_scaled = scaler.transform(df)

    return df_scaled
