
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# --- Load data from local 'data' folder ---
data_dir = "data"

frp_file     = os.path.join(data_dir, "Kilauea_Daily_FRP.csv")
seis_file    = os.path.join(data_dir, "Kilauea_Daily_Seismic.csv")
so2_file     = os.path.join(data_dir, "Kilauea_Daily_SO2.csv")

df_frp  = pd.read_csv(frp_file, parse_dates=["Date"])
df_seis = pd.read_csv(seis_file, parse_dates=["Date"])
df_so2  = pd.read_csv(so2_file, parse_dates=["Date"])

# --- Merge datasets on Date ---
df = (
    df_frp.merge(df_seis, on="Date", how="inner")
          .merge(df_so2,  on="Date", how="inner")
)

# Standardise column names for consistency
df = df.rename(columns={
    "frp": "FRP",
    "NosQuakes": "NosQuakes",
    "MeanMagQuakes": "MeanMag",
    "SO2": "SO2"
})

# --- Target = FRP next day ---
df["FRP_tomorrow"] = df["FRP"].shift(-1)

# Drop last row (no target available)
df = df.dropna()

# --- Features (X) and target (y) ---
X = df[["NosQuakes", "MeanMag", "SO2", "FRP"]].values
y = df["FRP_tomorrow"].values

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print(df[["NosQuakes", "MeanMag", "SO2", "FRP", "FRP_tomorrow"]].head())

# --- Train/test split ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Random Forest model ---
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --- Evaluation ---
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))

# --- Feature importance plot ---
importances = model.feature_importances_
labels = ["NosQuakes", "MeanMag", "SO2", "FRP"]

plt.bar(labels, importances)
plt.title("Feature Importance for Predicting Next-Day FRP")
plt.show()


