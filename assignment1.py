#  Exponential Smoothing 

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#  1) Load
train_df = pd.read_csv("assignment_data_train.csv")
test_df  = pd.read_csv("assignment_data_test.csv")

# 2) Prepare data 
train_df["Timestamp"] = pd.to_datetime(train_df["Timestamp"], errors="coerce")
test_df["Timestamp"]  = pd.to_datetime(test_df["Timestamp"], errors="coerce")

train_df = train_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")
test_df  = test_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")

train_y = pd.to_numeric(train_df["trips"], errors="coerce")
train_y = train_y.interpolate(method="time").ffill().bfill()

# 3) Train model 
model = ExponentialSmoothing(
    train_y,
    trend="add",
    seasonal="add",
    seasonal_periods=24,
    initialization_method="estimated"
)
modelFit = model.fit(optimized=True, use_brute=True)

# 4) Forecast
h = len(test_df)
forecast = modelFit.forecast(steps=h)


pred = np.maximum(np.asarray(forecast, dtype=float).ravel(), 0.0)

#  Debug 
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Prediction length:", len(pred))
print("First 5 predictions:", pred[:5])










