#  Exponential Smoothing 


import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1) Load data 
TRAIN_FILE = "assignment_data_train.csv"
TEST_FILE  = "assignment_data_test.csv"

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Timestamp"])

train_df = _read_csv(TRAIN_FILE).sort_values("Timestamp").set_index("Timestamp")
test_df  = _read_csv(TEST_FILE).sort_values("Timestamp").set_index("Timestamp")

# 2) Prepare target (hourly, numeric, no gaps)
target_col = "SystemLoadEA"

# coerce to numeric, set hourly freq, fill gaps by time interpolation
train_y = (
    pd.to_numeric(train_df[target_col], errors="coerce")
      .asfreq("h")
      .interpolate(method="time", limit_direction="both")
      .values
)

# Forecast horizon = length of test set
h = len(test_df)

# 3) Build & fit model
model = ExponentialSmoothing(
    train_y,
    trend="add",
    seasonal="add",
    seasonal_periods=24,
    initialization_method="estimated",
)
modelFit = model.fit(optimized=True, use_brute=True)

# 4) Forecast
forecast = modelFit.forecast(steps=h)
pred = np.asarray(forecast, dtype=float).ravel()
pred = np.maximum(pred, 0.0)  # ensure non-negative








