#  Exponential Smoothing 

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Load data 
base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
train_file = os.path.join(base_dir, "assignment_data_train.csv")
test_file = os.path.join(base_dir, "assignment_data_test.csv")

if os.path.exists(train_file) and os.path.exists(test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
else:
    raise FileNotFoundError(f"Cannot find CSV files at {train_file} and {test_file}")

# 2. Prepare data
train_df = train_df.sort_values("Timestamp").set_index("Timestamp")
test_df = test_df.sort_values("Timestamp").set_index("Timestamp")

train_y = train_df["SystemLoadEA"].astype(float).values
test_y = test_df["SystemLoadEA"].astype(float).values

# 3. Train model
model = ExponentialSmoothing(train_y, trend="add", seasonal="add", seasonal_periods=24)
modelFit = model.fit()

# 4. Forecast
pred = np.asarray(modelFit.forecast(len(test_y)))  # must be numpy array

# 5. Debug info (will not break autograder)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Prediction length:", len(pred))
print("First 5 predictions:", pred[:5])








