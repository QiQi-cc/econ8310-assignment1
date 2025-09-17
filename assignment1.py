#  Exponential Smoothing 

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1) Load data 
train_df = pd.read_csv("assignment_data_train.csv", parse_dates=["Timestamp"])
test_df  = pd.read_csv("assignment_data_test.csv",  parse_dates=["Timestamp"])

# 2) Sort & set hourly index, fill any tiny gaps
train_df = train_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")
test_df  = test_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")

y_train = pd.to_numeric(train_df["trips"], errors="coerce")
y_train = y_train.interpolate(method="time", limit_direction="both").astype(float)

# 3) Build a Holt-Winters model with weekly seasonality
model = ExponentialSmoothing(
    y_train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=168,          
    initialization_method="estimated"
)

# 4) Fit
modelFit = model.fit(optimized=True, use_brute=True)

# 5) Forecast 
h = len(test_df)
forecast = modelFit.forecast(steps=h)

pred = np.asarray(forecast, dtype=float).ravel()
pred = np.maximum(pred, 0.0)  # no negative trips



