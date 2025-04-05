# predict.py
import numpy as np
import pickle
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "water_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict_water_potability(features):
    """
    Predict water potability based on user input.

    Args:
        features (list or array): List of 9 numeric features in order:
            [ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
            Organic_carbon, Trihalomethanes, Turbidity]

    Returns:
        int: 0 (not potable) or 1 (potable)
    """
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction
