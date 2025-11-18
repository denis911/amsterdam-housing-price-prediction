# trains model and saves artifact
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error
import pickle


# STEP 0 - download dataset
df = pd.read_csv("HousingPrices-Amsterdam-August-2021.csv")
df.columns = df.columns.str.lower()

df['pc4'] = df.zip.str.split().str[0]
used_cols = ['pc4', 'area', 'room', 'price']
df = df[used_cols]

# STEP 1 - prepare X_full_train
# make number of rooms and area as integers
df['area'] = df['area'].astype('int64')
df['room'] = df['room'].astype('int64')

# Delete 4 rows with missing price out of 924
df = df.dropna(subset=['price'])
df['price'] = df['price'].astype('int64')

# log1p price column
df.price = np.log1p(df.price)

# Do full_train/test split with 80%/20% distribution.
# Use the train_test_split function and set the random_state parameter to 1.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# reset index in splits
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# prepare label (target variable)
y_full_train = df_full_train.price.values
y_test = df_test.price.values

# remove label so model cannot learn from it accidentally
del df_full_train['price']
del df_test['price']

# we use many categorical features, thus one-hot encoding is needed:
full_train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

# Use DictVectorizer(sparse=True) to turn the dataframes into matrices.
dv = DictVectorizer(sparse=True)
X_full_train = dv.fit_transform(full_train_dicts) 
X_test = dv.transform(test_dicts) 

features = list(dv.get_feature_names_out())

# Build dmatrices:
d_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# STEP 2 - train XGBoost model
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, d_full_train, num_boost_round=100)

y_pred = model.predict(dtest)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}") 

# On full train we sould get RMSE: 0.1948
# Practically we got RMSE: 0.1922

# STEP 3 - pickle model and dv together

#  dv = trained DictVectorizer
#  model = trained XGBoost Booster model

data_to_save = {
    "dv": dv,
    "model": model
}

with open("Ams_xgb_pipeline.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

