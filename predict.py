# predict.py - load a saved model and predict on new rows
import joblib
import pandas as pd
import os

MODEL_PATH = "models/model.joblib"

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict(df):
    model = load_model()
    return model.predict(df)

if __name__ == "__main__":
    # Example usage - replace with real input
    sample = pd.read_csv("data/sample.csv").drop(columns=["target"]).head(5)
    print(predict(sample))
