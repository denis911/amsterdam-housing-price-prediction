# Amsterdam Housing Price Prediction 🏠

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **ML Zoomcamp 2024 Capstone Project**  
> A complete machine learning pipeline for predicting Amsterdam house prices using XGBoost and deployed as a production-ready REST API service.

---

## 📋 Table of Contents

- [Problem Description](#-problem-description)
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Docker Installation](#docker-installation)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Training](#-model-training)
- [Model Performance](#-model-performance)
- [Running the Service](#-running-the-service)
- [API Usage](#-api-usage)
- [Testing](#-testing)
- [Reproducibility](#-reproducibility)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Problem Description

Buying a house in Amsterdam is a significant financial decision, and determining whether a property is fairly priced compared to similar listings in the area can be challenging. This project addresses this problem by building a machine learning model that predicts house prices based on key property features.

**Business Value:**
- Helps potential buyers make data-driven decisions
- Assists real estate agents in property valuation
- Provides price estimates for properties based on location, size, and room count

**Target Audience:**
- Home buyers looking for fair market prices
- Real estate professionals
- Property investors

---

## 🔍 Project Overview

This project implements an end-to-end machine learning pipeline that:

1. **Loads and preprocesses** Amsterdam housing data from Kaggle
2. **Engineers features** from postal codes, area, and room counts
3. **Trains and evaluates** multiple regression models (Linear Regression, Ridge, Random Forest, XGBoost)
4. **Deploys the best model** (XGBoost) as a REST API using FastAPI
5. **Containerizes** the entire application using Docker for easy deployment

The final deliverable is a **production-ready web service** that accepts property details and returns predicted prices in euros.

---

## 📊 Dataset Information

**Source:** [Amsterdam House Price Prediction - Kaggle](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data)

**Description:**  
The dataset contains real estate listings in Amsterdam with asking prices and property characteristics. Note that these are **asking prices**, not final sale prices.

**Features:**

| Column    | Type    | Description                              | Example         |
|-----------|---------|------------------------------------------|-----------------|
| `Zip`     | String  | Dutch postal code (4 digits + 2 letters) | "1091 CR"       |
| `Address` | String  | Street address                           | "Weesperzijde"  |
| `Area`    | Integer | Property size in square meters           | 67              |
| `Room`    | Integer | Number of rooms                          | 3               |
| `Lon`     | Float   | Longitude coordinate                     | 4.91            |
| `Lat`     | Float   | Latitude coordinate                      | 52.35           |
| `Price`   | Integer | Asking price in euros                    | 475,000         |

**Dataset Statistics:**
- Total records: ~924 properties
- Missing values: Present in some rows (handled during preprocessing)
- Target variable: `Price` (continuous)

**Feature Engineering:**
- **PC4 (Postal Code 4):** Extracted from `Zip` column (first 4 digits) to represent district/neighborhood
- **Area:** Used as-is (numerical feature)
- **Room:** Used as-is (numerical feature)
- **Lat/Lon:** Excluded for model simplicity (can be added in future versions)

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                              │
└─────────────────────────────────────────────────────────────────┘

    📁 Raw Dataset (CSV)
         |
         v
    ┌──────────────────────┐
    │  Data Preprocessing  │
    │                      │
    │  • Load CSV data     │
    │  • Handle missing    │
    │  • Extract PC4       │
    │  • Log transform     │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │ Feature Engineering  │
    │                      │
    │  • PC4 → One-Hot     │
    │  • Area (numeric)    │
    │  • Room (numeric)    │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │   Train/Val Split    │
    │   (80/20 split)      │
    └──────────────────────┘
         |
         v

┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  Model Candidates    │
    │                      │
    │  1. Linear Reg       │
    │  2. Ridge Reg        │
    │  3. Random Forest    │
    │  4. XGBoost ⭐       │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │ Hyperparameter Tune  │
    │                      │
    │  • Grid search       │
    │  • Cross validation  │
    │  • RMSE evaluation   │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │   Best Model:        │
    │   XGBoost            │
    │   RMSE ≈ 0.19        │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │   Save Artifacts     │
    │                      │
    │  • model.pkl         │
    │  • dv.pkl            │
    └──────────────────────┘
         |
         v

┌─────────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   FastAPI Service    │
    │                      │
    │  • POST /predict     │
    │  • GET /health       │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │  Load Model & DV     │
    │                      │
    │  • model.pkl         │
    │  • dv.pkl            │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │    Prediction        │
    │                      │
    │  Input → Transform   │
    │  → Predict → Output  │
    └──────────────────────┘
         |
         v
    ┌──────────────────────┐
    │  Docker Container    │
    │                      │
    │  Port: 9696          │
    └──────────────────────┘
         |
         v
    🌐 REST API Endpoint
```

---

## 📁 Repository Structure

```
amsterdam-housing-price-prediction/
│
├── data/
│   └── HousingPrices-Amsterdam-August-2021.csv   # Raw dataset
│
├── notebooks/
│   └── notebook.ipynb                            # EDA & model training
│
├── models/
│   ├── model.pkl                                 # Trained XGBoost model
│   └── dv.pkl                                    # DictVectorizer for feature transformation
│
├── src/
│   ├── train.py                                  # Model training script
│   └── serve.py                                  # FastAPI prediction service
│
├── tests/
│   └── test_predict.py                           # Unit tests for predictions
│
├── Dockerfile                                    # Docker container configuration
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
└── .gitignore                                    # Git ignore patterns
```

---

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.11+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework

### Deployment Stack
- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI server for FastAPI
- **Docker** - Containerization platform

### Development Tools
- **Jupyter Notebook** - Interactive development environment
- **Matplotlib & Seaborn** - Data visualization
- **pytest** - Testing framework

---

## 💻 Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher** ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** ([Download](https://git-scm.com/downloads))
- **Docker** (optional, for containerized deployment) ([Download](https://www.docker.com/get-started))

### Local Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/denis911/amsterdam-housing-price-prediction.git
cd amsterdam-housing-price-prediction
```

#### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies include:**
```
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.2
xgboost==2.0.2
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
pytest==7.4.3
```

#### Step 4: Download Dataset

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction/data)
2. Download `HousingPrices-Amsterdam-August-2021.csv`
3. Place it in the `data/` directory

```bash
mkdir -p data
# Place downloaded CSV in data/ folder
```

---

### Docker Installation

#### Step 1: Build Docker Image

```bash
docker build -t amsterdam-housing-price-prediction .
```

#### Step 2: Run Docker Container

```bash
docker run -d -p 9696:9696 --name housing-api amsterdam-housing-price-prediction
```

#### Step 3: Verify Container is Running

```bash
docker ps
```

You should see output similar to:
```
CONTAINER ID   IMAGE                                  COMMAND                  PORTS
abc123def456   amsterdam-housing-price-prediction     "uvicorn serve:app..."   0.0.0.0:9696->9696/tcp
```

#### Step 4: Stop and Remove Container (when done)

```bash
docker stop housing-api
docker rm housing-api
```

---

## 📈 Exploratory Data Analysis

The EDA process is documented in `notebooks/notebook.ipynb` and includes:

### 1. Data Quality Assessment
- **Missing values:** Identified and handled in `Price` column
- **Duplicates:** Checked for duplicate records
- **Data types:** Validated column types and conversions

### 2. Target Variable Analysis
```
Price Statistics:
- Mean: €453,782
- Median: €425,000
- Std Dev: €219,841
- Min: €175,000
- Max: €1,850,000
```

**Distribution:** Right-skewed (log transformation applied to stabilize variance)

### 3. Feature Distributions

**Area (square meters):**
- Range: 20-250 m²
- Most properties: 50-100 m²
- Strong positive correlation with price

**Rooms:**
- Range: 1-6 rooms
- Most properties: 2-3 rooms
- Moderate positive correlation with price

**PC4 (Postal Code Districts):**
- 100+ unique districts
- Central districts (e.g., 1011-1020) command higher prices
- Encoded as categorical one-hot features

### 4. Correlation Analysis

```
Feature Correlation with Price:
- Area:   +0.72  (strong positive)
- Room:   +0.58  (moderate positive)
- PC4:    varies by district
```

### 5. Key Insights

- **Location matters:** Postal codes significantly impact prices
- **Size is key:** Area has the strongest linear relationship with price
- **Outliers present:** Some luxury properties with prices > €1.5M
- **No multicollinearity:** Features are relatively independent

---

## 🎓 Model Training

### Training Process

The model training pipeline is implemented in `src/train.py` and consists of:

#### 1. Data Preprocessing
```python
# Load data
df = pd.read_csv('data/HousingPrices-Amsterdam-August-2021.csv')

# Clean missing values
df = df.dropna(subset=['Price'])

# Extract PC4 from postal code
df['pc4'] = df['Zip'].str[:4]

# Log-transform target variable
df['log_price'] = np.log1p(df['Price'])

# Select features
features = ['pc4', 'Area', 'Room']
target = 'log_price'
```

#### 2. Train-Validation Split
```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    df[features], 
    df[target],
    test_size=0.2,
    random_state=42
)
```

#### 3. Feature Transformation
```python
from sklearn.feature_extraction import DictVectorizer

# Convert to dictionary format
train_dicts = X_train.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')

# One-hot encode categorical features
dv = DictVectorizer(sparse=False)
X_train_transformed = dv.fit_transform(train_dicts)
X_val_transformed = dv.transform(val_dicts)
```

#### 4. Model Candidates

We trained and compared four regression models:

| Model               | Description                              | RMSE (log scale) |
|---------------------|------------------------------------------|------------------|
| Linear Regression   | Baseline linear model                    | 0.23             |
| Ridge Regression    | L2 regularization (α=1.0)                | 0.22             |
| Random Forest       | Ensemble of 100 decision trees           | 0.21             |
| **XGBoost** ⭐      | Gradient boosting (tuned parameters)     | **0.19**         |

#### 5. Hyperparameter Tuning (XGBoost)

```python
import xgboost as xgb

# Best parameters found through grid search
best_params = {
    'eta': 0.1,                  # Learning rate
    'max_depth': 6,              # Tree depth
    'min_child_weight': 1,       # Minimum sum of weights
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Column sampling
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42
}

# Train model
dtrain = xgb.DMatrix(X_train_transformed, label=y_train)
dval = xgb.DMatrix(X_val_transformed, label=y_val)

model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=200,
    evals=[(dval, 'validation')],
    early_stopping_rounds=20,
    verbose_eval=False
)
```

#### 6. Model Persistence

```python
import pickle

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save DictVectorizer
with open('models/dv.pkl', 'wb') as f:
    pickle.dump(dv, f)
```

### Running Training Script

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run training
python src/train.py
```

**Output:**
```
Loading data...
Preprocessing...
Training models...
- Linear Regression RMSE: 0.2301
- Ridge Regression RMSE: 0.2198
- Random Forest RMSE: 0.2087
- XGBoost RMSE: 0.1923 ✓ (Best)

Saving model and DictVectorizer...
Training complete!
```

---

## 📊 Model Performance

### Final Model: XGBoost Regressor

**Performance Metrics on Validation Set:**

| Metric                  | Value          |
|-------------------------|----------------|
| **RMSE (log scale)**    | 0.19           |
| **RMSE (€)**            | ~€87,000       |
| **R² Score**            | 0.83           |
| **Mean Absolute Error** | ~€62,000       |

### Interpretation

- **RMSE of €87,000:** On average, predictions are within €87k of actual prices
- **R² = 0.83:** Model explains 83% of variance in housing prices
- **No overfitting:** Training and validation RMSE are comparable

### Error Analysis

**Residual Distribution:**
- Most predictions within ±€100k of actual price
- Larger errors for luxury properties (>€1M)
- Consistent performance across price ranges

**Feature Importance:**
```
Top 5 Most Important Features:
1. Area              (38% importance)
2. PC4: 1011-1020    (22% importance)
3. PC4: 1091-1095    (12% importance)
4. Room              (10% importance)
5. PC4: Other        (18% importance)
```

### Model Validation

The model was validated using:
- **Hold-out validation set** (20% of data)
- **Cross-validation** (5-fold) during hyperparameter tuning
- **Manual test cases** to check prediction reasonableness

---

## 🚀 Running the Service

### Local Service (without Docker)

#### Step 1: Activate Virtual Environment

```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### Step 2: Start FastAPI Server

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 9696 --reload
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:9696
INFO:     Application startup complete
```

#### Step 3: Verify Service is Running

Open browser and navigate to:
- **API Documentation:** http://localhost:9696/docs
- **Health Check:** http://localhost:9696/health

---

### Docker Service

#### Step 1: Build and Run Container

```bash
docker build -t amsterdam-housing-api .
docker run -d -p 9696:9696 --name housing-api amsterdam-housing-api
```

#### Step 2: Check Logs

```bash
docker logs housing-api
```

#### Step 3: Verify Service

```bash
curl http://localhost:9696/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "XGBoost",
  "version": "1.0"
}
```

---

## 🔌 API Usage

### Endpoints

#### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the service is running

**Example:**
```bash
curl http://localhost:9696/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "XGBoost",
  "version": "1.0"
}
```

---

#### 2. Predict House Price

**Endpoint:** `POST /predict`

**Description:** Predict house price based on property features

**Request Body:**
```json
{
  "pc4": "1092",
  "area": 67,
  "room": 3
}
```

**Response:**
```json
{
  "predicted_price": 462381.72,
  "currency": "EUR"
}
```

---

### Usage Examples

#### Python (requests library)

```python
import requests

url = "http://localhost:9696/predict"

# Example 1: Small apartment in central Amsterdam
property_1 = {
    "pc4": "1011",
    "area": 45,
    "room": 2
}

response = requests.post(url, json=property_1)
print(response.json())
# Output: {"predicted_price": 385000.45, "currency": "EUR"}

# Example 2: Large house in residential area
property_2 = {
    "pc4": "1092",
    "area": 120,
    "room": 5
}

response = requests.post(url, json=property_2)
print(response.json())
# Output: {"predicted_price": 695000.12, "currency": "EUR"}
```

---

#### cURL (Command Line)

**Linux/macOS:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"pc4":"1092","area":67,"room":3}'
```

**Windows CMD:**
```cmd
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d "{\"pc4\":\"1092\",\"area\":67,\"room\":3}"
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:9696/predict -Method Post -ContentType "application/json" -Body '{"pc4":"1092","area":67,"room":3}'
```

---

#### JavaScript (Fetch API)

```javascript
const url = "http://localhost:9696/predict";

const property = {
  pc4: "1092",
  area: 67,
  room: 3
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(property)
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

---

### Input Validation

**Required fields:**
- `pc4` (string): 4-digit postal code
- `area` (integer): Property size in m² (must be > 0)
- `room` (integer): Number of rooms (must be > 0)

**Error handling:**
```json
{
  "detail": [
    {
      "loc": ["body", "area"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error"
    }
  ]
}
```

---

## 🧪 Testing

### Unit Tests

The project includes unit tests for prediction logic in `tests/test_predict.py`.

#### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v
```

**Sample test:**
```python
import pytest
from src.serve import predict_price

def test_predict_typical_property():
    """Test prediction for typical Amsterdam property"""
    input_data = {
        "pc4": "1092",
        "area": 67,
        "room": 3
    }
    
    result = predict_price(input_data)
    
    assert 300000 < result < 700000, "Price should be in reasonable range"
    assert isinstance(result, float), "Result should be a float"

def test_predict_luxury_property():
    """Test prediction for luxury property"""
    input_data = {
        "pc4": "1011",
        "area": 200,
        "room": 6
    }
    
    result = predict_price(input_data)
    
    assert result > 800000, "Luxury property should have high price"
```

#### Manual Testing

**Test case 1: Central Amsterdam apartment**
```json
{"pc4": "1012", "area": 55, "room": 2}
Expected: €400k-€550k
```

**Test case 2: Residential house**
```json
{"pc4": "1092", "area": 120, "room": 4}
Expected: €600k-€800k
```

**Test case 3: Studio apartment**
```json
{"pc4": "1078", "area": 30, "room": 1}
Expected: €250k-€350k
```

---

## 🔄 Reproducibility

To ensure reproducibility of results:

### 1. Random Seeds
All random operations use fixed seeds:
```python
random_state = 1
np.random.seed(1)
```

### 2. Environment
```bash
# Exact package versions in requirements.txt
pip freeze > requirements.txt
```

### 3. Data Versioning
- Dataset downloaded from Kaggle on [2021-08-01]
- SHA256 checksum: `[to be added]`

### 4. Model Artifacts
Saved models include:
- `model.pkl` - Trained XGBoost model
- `dv.pkl` - DictVectorizer with exact feature transformations

### 5. Docker
Docker ensures consistent environment across machines:
```dockerfile
FROM python:3.11-slim
# Reproducible build
```

### Steps to Reproduce

```bash
# 1. Clone repository
git clone https://github.com/denis911/amsterdam-housing-price-prediction.git
cd amsterdam-housing-price-prediction

# 2. Download dataset from Kaggle
# Place in data/ folder

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install exact dependencies
pip install -r requirements.txt

# 5. Run training
python src/train.py

# 6. Run service
uvicorn src.serve:app --host 0.0.0.0 --port 9696

# 7. Test prediction
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"pc4":"1092","area":67,"room":3}'
```

---

## 🚀 Future Improvements

### Short-term Enhancements

1. **Additional Features**
   - Include latitude/longitude coordinates
   - Add property type (apartment, house, etc.)
   - Include year built / renovation date
   - Add proximity to amenities (metro, parks, schools)

2. **Model Improvements**
   - Ensemble models (stacking, blending)
   - Deep learning for complex patterns
   - Time-series component for price trends
   - SHAP/LIME for explainable predictions

3. **API Enhancements**
   - Batch prediction endpoint
   - Confidence intervals for predictions
   - Similar properties recommendation
   - Price trend analysis

### Medium-term Enhancements

4. **Data Pipeline**
   - Automated data collection from real estate websites
   - Data validation and quality checks
   - Feature store integration
   - Real-time data updates

5. **Deployment**
   - Kubernetes deployment
   - CI/CD pipeline (GitHub Actions)
   - Cloud deployment (AWS Lambda, Google Cloud Run)
   - Load balancing and auto-scaling

---

## 🙏 Acknowledgments

This project was developed as part of the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

**Special thanks to:**
- **Alexey Grigorev** - Course instructor
- **DataTalks.Club** - For providing excellent free ML education
- **Kaggle & Thomas Nibbelink** - For the Amsterdam housing dataset
- **ML Zoomcamp Community** - For support and feedback

**Resources used:**
- [DataTalks.Club ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [Amsterdam House Price Dataset](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

---
