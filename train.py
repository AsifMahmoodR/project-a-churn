# train.py - minimal training flow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/sample.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Placeholder - replace with real steps
    df = df.dropna().copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    train()
