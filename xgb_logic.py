import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def train_and_evaluate(data):  # Make sure the parameter is named 'data'
    
    # 1. Define your target and columns to drop
    target_col = 'PM10,24HOUR' if 'PM10,24HOUR' in data.columns else 'PM10'
    drop_cols = ['Year', 'Date', 'Time', 'Datetime', target_col]
    
    # 2. THE FIX: Change 'df' to 'data' here so it uses the passed-in dataframe!
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)
    
    # Calculate advanced metrics
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'Bias': np.sum(y_pred - y_test_arr) / len(y_test_arr),
        'IA': 1 - (np.sum((y_pred - y_test_arr)**2) / np.sum((np.abs(y_pred - np.mean(y_test_arr)) + np.abs(y_test_arr - np.mean(y_test_arr)))**2))
    }
    
    return model, metrics, X_test, y_test, y_pred