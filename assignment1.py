#  Exponential Smoothing 

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE = "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/"
TRAIN_URL = BASE + "assignment_data_train.csv"
TEST_URL  = BASE + "assignment_data_test.csv"

# ---- Load data ----
train_df = pd.read_csv(TRAIN_URL, parse_dates=["Timestamp"])
test_df  = pd.read_csv(TEST_URL,  parse_dates=["Timestamp"])


train_df = train_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")
test_df  = test_df.sort_values("Timestamp").set_index("Timestamp").asfreq("h")

# ---- Prepare target series ----
y = pd.to_numeric(train_df["trips"], errors="coerce")
y = y.interpolate(method="time").ffill().bfill()
y = y.clip(lower=1e-6)  # multiplicative seasonal needs strictly positive values

# ---- Build the model ----
model = ExponentialSmoothing(
    y,
    trend="add",
    damped_trend=True,
    seasonal="mul",
    seasonal_periods=24,
    initialization_method="estimated",
)

modelFit = model.fit(
    optimized=True,
    use_brute=True,
    use_boxcox=True,
    remove_bias=True,
)

# ---- Forecast the exact length of the test set ----
h = len(test_df)
pred = modelFit.forecast(h).astype(float).ravel()
pred = np.maximum(pred, 0.0)  # grader requires nonnegative values

# Optional debug when running locally
if __name__ == "__main__":
    print(f"Train len: {len(y)}, Test len: {h}")
    print("First 5 predictions:", pred[:5])


