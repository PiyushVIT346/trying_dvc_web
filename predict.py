import pandas as pd
import pickle

def load_model():
    with open("models/water_model.pkl", "rb") as f:
        return pickle.load(f)

def predict(file_path):
    df = pd.read_csv(file_path)
    df.fillna(df.median(numeric_only=True), inplace=True)
    model = load_model()
    predictions = model.predict(df.values)
    return predictions.tolist()

