# loads model, runs inference
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# helper function to return price from values entered
def predict_price(pc4, area, room, dv, model):
    # 1. Create a record exactly like training data
    record = {
        "pc4": str(pc4),   # pc4 is categorical → string
        "area": float(area),
        "room": float(room)
    }

    # 2. Transform with DictVectorizer
    X = dv.transform([record])

    # 3. Predict (log scale)
    dmatrix = xgb.DMatrix(X, feature_names=dv.feature_names_)
    y_log = model.predict(dmatrix)[0]

    # 4. Convert log1p → price
    return np.expm1(y_log)


# Load saved model from local drive
with open("Ams_xgb_pipeline.pkl", "rb") as f:
    loaded = pickle.load(f)

dv = loaded["dv"]
model = loaded["model"]

# Hard code values from row 100 of dataset to test predictions
pc4  = '1092'
area = 67
room = 3
real = 500000.0

pred = predict_price(pc4, area, room, dv, model)

print("Real price:     ", real)
print("Predicted price:", pred)
print("Error:          ", pred - real)
# Error: -42163.1875 - about 8%
