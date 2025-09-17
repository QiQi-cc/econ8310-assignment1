#  Exponential Smoothing 

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Load data
train_df = pd.read_csv("assignment_data_train.csv")
test_df = pd.read_csv("assignment_data_test.csv")

# 2. Select target column
y_train = train_df["system_load"]

# 3. Build model
model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

# 4. Fit model
modelFit = model.fit()

# 5. Make predictions 
pred = modelFit.forecast(len(test_df))




