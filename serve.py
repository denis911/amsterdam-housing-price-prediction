# starts web service
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import xgboost as xgb
import numpy as np

# ---------------------------------------------------
# Load model + dict vectorizer
# ---------------------------------------------------
with open("Ams_xgb_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

dv = pipeline["dv"]
model = pipeline["model"]

# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------
app = FastAPI(title="Amsterdam House Price Predictor")

# ---------------------------------------------------
# Input schema
# ---------------------------------------------------
class HouseInput(BaseModel):
    pc4: str
    area: int
    room: int

# ---------------------------------------------------
# Helper prediction function
# ---------------------------------------------------
def predict_price(input: HouseInput):
    record = {
        "pc4": input.pc4,
        "area": input.area,
        "room": input.room
    }

    X = dv.transform([record])
    dmatrix = xgb.DMatrix(X, feature_names=dv.feature_names_)
    y_log = model.predict(dmatrix)[0]

    return float(np.expm1(y_log))   # convert back from log1p


# ---------------------------------------------------
# Endpoints
# ---------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: HouseInput):
    price = predict_price(input)
    return {"predicted_price": price}

# Endpoint addresses:
# http://localhost:9696/predict
# http://localhost:9696/health

# How to run prediction as a local web service:
# uv run uvicorn serve:app --reload --host 0.0.0.0 --port 9696

# Swagger UI: http://localhost:9696/docs

# Example JSON POST request
# {
#   "pc4": "1092",
#   "area": 67,
#   "room": 3
# }
# It corresponds to row 100 from the dataset, price should be around 500_000.0

# Windows CMD CURL that works:
# curl -X POST -H "Content-Type: application/json" -d "{\"pc4\":\"1092\", \"area\":67, \"room\":3}" http://localhost:9696/predict

# You should receive something like:
# {
#   "predicted_price": 462381.72
# }

