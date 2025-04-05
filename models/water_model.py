import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from dvclive import Live
import yaml

# Load hyperparameters
n_estimators = yaml.safe_load(open('D:/projects/MERN AI hackathon2/dvc_trying/dvclive/params.yaml'))["n_estimators"]

# Load and split dataset
data = pd.read_csv("data/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Fill missing values with median
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Prepare training and testing sets
x_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values

x_test = test_processed_data.iloc[:, :-1].values
y_test = test_processed_data.iloc[:, -1].values

# Train and save model
clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
clf.fit(x_train, y_train)

with open("water_model.pkl", 'wb') as f:
    pickle.dump(clf, f)

# Load model and evaluate
with open("water_model.pkl", 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(x_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log with DVC Live
with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy", acc)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1 score", f1)
    live.log_param("n_estimators", n_estimators)
