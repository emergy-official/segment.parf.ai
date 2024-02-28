import os

import joblib
import pandas as pd
from xgboost import XGBClassifier

def load_model(model_path: str) -> XGBClassifier:
    """
    Load the model from the specified directory.
    """
    return joblib.load(model_path)


def predict(body: dict, model: XGBClassifier) -> dict:
    """
    Generate predictions for the incoming request using the model.
    """
    features = pd.DataFrame.from_records(body["data"])
    predictions = model.predict(features).tolist()
    return {"predictions": predictions}