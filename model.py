# ml_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Load dataset
df = pd.read_excel("ENB2012_data.xlsx")  # Replace with your file name

# Features and targets
X = df.iloc[:, :-2]  # First 8 columns
y = df.iloc[:, -2:]  # Last 2 columns: Y1 and Y2

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("📊 Model Evaluation:")
print("Heating Load (Y1):")
print(" - RMSE:", np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0])))
print(" - R² Score:", r2_score(y_test.iloc[:, 0], y_pred[:, 0]))

print("\nCooling Load (Y2):")
print(" - RMSE:", np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1])))
print(" - R² Score:", r2_score(y_test.iloc[:, 1], y_pred[:, 1]))

# Save model
joblib.dump(model, "energy_eff_model.pkl")
print("\n✅ Model saved as 'energy_eff_model.pkl'")
