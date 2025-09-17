#  Exponential Smoothing 

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Load data 
train_file = "assignment_data_train.csv"
test_file = "assignment_data_test.csv"

if os.path.exists(train_file) and os.path.exists(test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
elif os.path.exists("../" + train_file) and os.path.exists("../" + test_file):
    train_df = pd.read_csv("../" + train_file)
    test_df = pd.read_csv("../" + test_file)
else:
    raise FileNotFoundError("Cannot find training/testing CSV files in current or parent directory.")

# 2. Prepare data
train_df = train_df.sort_values("Timestamp").set_index("Timestamp")
test_df = test_df.sort_values("Timestamp").set_index("Timestamp")

# Ensure numeric values
train_y = train_df["SystemLoadEA"].astype(float).values
test_y = test_df["SystemLoadEA"].astype(float).values

# 3. Train model
model = ExponentialSmoothing(train_y, trend="add", seasonal="add", seasonal_periods=24)
modelFit = model.fit()

# 4. Forecast
pred = modelFit.forecast(len(test_y))

# 5. Debug info
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Prediction length:", len(pred))
print("First 5 predictions:", pred[:5])






