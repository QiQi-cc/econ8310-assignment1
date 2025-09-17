import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data
train_df = pd.read_csv("assignment_data_train.csv", parse_dates=["Timestamp"])
test_df = pd.read_csv("assignment_data_test.csv", parse_dates=["Timestamp"])

# Preprocess
train_df = train_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")
y_train = pd.to_numeric(train_df["trips"], errors="coerce").interpolate(method="time")

# Define the model 
model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=24,
    initialization_method="estimated"
)

# Fit the model
modelFit = model.fit(optimized=True, use_brute=True)

# Forecast
h = len(test_df)
forecast = modelFit.forecast(steps=h)

pred = np.asarray(forecast, dtype=float).ravel()
pred = np.maximum(pred, 0)
