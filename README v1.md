# github: amsterdam-housing-price-prediction
(Took small Kaggle dataset, built 3 models, selected 1 and deployed it.)

# TLDR: Amsterdam House Price Prediction - full project description

📍 GitHub Repo: https://github.com/denis911/amsterdam-housing-price-prediction  
📊 Dataset: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data  

---

## Table of Contents

1. Project Overview
2. Dataset Description
3. Project Architecture
4. Environment Setup
5. Data Preparation
6. Model Training
7. Prediction Service (FastAPI)
8. Dockerization
9. Usage Examples
10. Future Improvements
11. References

---

## 1. Project Overview

This project is a **machine learning pipeline** that predicts Amsterdam house prices based on publicly available real estate data.  
The pipeline includes:

- Data cleaning and preprocessing
- Feature engineering (postcode, area, rooms)
- Model training with **XGBoost**
- Deployment via **FastAPI**  
- Containerization with **Docker**  

Goal: Provide a **ready-to-use web service** that can predict house prices based on user input.

---

## 2. Dataset Description

The dataset is from Kaggle: "Amsterdam House Price Prediction"  
Link: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data  

**Columns**:

| Column | Description |
|--------|-------------|
| zip    | Dutch postal code |
| address | Street address |
| area   | Size in square meters |
| room   | Number of rooms |
| price  | House price in euros |

**Notes:**

- Only numerical and categorical data are used. GPS location has been removed for simplicity.
- PC4 (first 4 digits of the postal code) is used as a categorical feature for districts.

---

## 3. Project Architecture

+------------------+
| CSV Dataset |
+------------------+
|
v
+------------------+
| Data Preprocessing|
| - clean missing |
| - extract PC4 |
| - one-hot encode |
+------------------+
|
v
+------------------+
| Model Training |
| - Linear Regression |
| - Ridge Regression |
| - Random Forest |
| - XGBoost |
+------------------+
|
v
+------------------+
| Prediction Pipeline |
| - DictVectorizer |
| - XGBoost Model |
+------------------+
|
v
+------------------+
| FastAPI Web Service |
| POST /predict |
| GET /health |
+------------------+
|
v
+------------------+
| Docker Container |
+------------------+


---

## 4. Environment Setup

### Requirements:

- Python 3.11+
- pip

Clone the repository:

git clone https://github.com/denis911/amsterdam-housing-price-prediction.git
cd amsterdam-housing-price-prediction


Create virtual environment:

python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows


Install dependencies:

pip install -r requirements.txt

---

## 5. Data Preparation

1. Load the CSV dataset.
2. Clean missing values (especially `price` column).
3. Extract **PC4** from postal code:

zip = "1091 CR" => pc4 = "1091"


4. One-hot encode categorical features:

- `pc4` → one-hot
- Numerical features remain as-is

5. Split dataset:

train_test_split(df, test_size=0.2, random_state=1)
train/validation split for hyperparameter tuning


---

## 6. Model Training

### Models used:

- **Linear Regression**
- **Ridge Regression**
- **Random Forest Regressior**
- **XGBoost Regressor**

### Steps:

1. Transform price with `np.log1p` to stabilize variance.
2. Use PC4 as categorical one-hot encoded feature.
3. Tune hyperparameters:

- Ridge: alpha regularization
- XGBoost: `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `num_boost_round`
- Evaluate RMSE on validation set

Example XGBoost tuning:

eta = 0.3
max_depth = [4,6,8]
min_child_weight = [1,5,10]


- Best model: XGBoost with RMSE ~0.19 (log-price) on test set
- No overfitting observed

---

## 7. Prediction Service (FastAPI)

FastAPI exposes endpoints:

GET /health -> returns {"status": "ok"}
POST /predict -> returns {"predicted_price": ...}


### Example usage:

#### JSON input

{
"pc4": "1092",
"area": 67,
"room": 3
}



#### Python request
import requests

url = "http://localhost:9696/predict"
payload = {"pc4":"1092","area":67,"room":3}

response = requests.post(url, json=payload)
print(response.json())


cURL (Linux / CMD)
curl -X POST -H "Content-Type: application/json" -d "{\"pc4\":\"1092\",\"area\":67,\"room\":3}" http://localhost:9696/predict

8. Dockerization
Dockerfile:

FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 9696
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]

Build and run

docker build -t ams-house-price .
docker run -it -p 9696:9696 ams-house-price

9. Usage Examples
Health check
GET http://localhost:9696/health
Response: {"status": "ok"}

Prediction
POST http://localhost:9696/predict
Body:
{
  "pc4": "1092",
  "area": 67,
  "room": 3
}
Response:
{
  "predicted_price": 462381.72
}

Testing multiple rows
Create a CSV with multiple rows

Transform with DictVectorizer

Send via POST in a loop or batch

10. Future Improvements
Add district sentiment feature using public APIs

Include Buurt/Wijk aggregation for feature engineering

Implement SHAP explanations for predictions

Add frontend UI (Streamlit or React)

Extend service for batch CSV prediction

11. References
Kaggle dataset: https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data

FastAPI: https://fastapi.tiangolo.com/

XGBoost: https://xgboost.readthedocs.io/

DataTalksClub Machine Learning Zoomcamp: https://github.com/DataTalksClub/ml-zoomcamp

ASCII Workflow Diagram

   CSV Dataset
        |
        v
  Data Preprocessing
  - clean missing
  - extract PC4
  - one-hot encode
        |
        v
   Model Training
  - Linear / Ridge / XGBoost
        |
        v
 Prediction Pipeline
  - DictVectorizer + XGBoost
        |
        v
  FastAPI Service
  POST /predict
  GET /health
        |
        v
 Docker Container

Author
Denis911
GitHub: https://github.com/denis911


---

This README covers:

- Dataset description  
- Detailed architecture with ASCII graphics  
- Step-by-step instructions to run locally  
- Docker instructions  
- Examples for Python and cURL  
- Suggestions for improvements  

---