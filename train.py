import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
           "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
FEATURES = ["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]

df = pd.read_csv("model_data/housing.csv", sep=r"\s+", header=None, names=COLUMNS)
# Drop a header row if the file has one (it would parse as non-numeric)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

X = df[FEATURES]
y = df["MEDV"]

model = Pipeline([
    ("scale", StandardScaler()),
    ("gbr", GradientBoostingRegressor(random_state=42)),
]).fit(X, y)

joblib.dump(model, "model_data/boston_housing_prediction.joblib")
print(f"Trained on {len(df)} rows. R^2 on training data: {model.score(X, y):.3f}")
