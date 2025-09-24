
# Loads daily Fire Radiative Power (FRP), seismic, and SO₂ data (2004-2025), merges them into one dataset, and trains a Random Forest model to predict tomorrow’s FRP using today’s values
# To run in Spyder for example, be in upper directory (volcanic-data-ai-application/) and run via: !python scripts\PredNextDay_FRP.py

# sklearn - machine learning library; ensamble - module which contains the RandomForestRegressor class
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load data from local 'data' folder 
data_dir = "data"

frp_file     = os.path.join(data_dir, "Kilauea_Daily_FRP.csv")
seis_file    = os.path.join(data_dir, "Kilauea_Daily_Seismic.csv")
so2_file     = os.path.join(data_dir, "Kilauea_Daily_SO2.csv")

df_frp  = pd.read_csv(frp_file, parse_dates=["Date"])
df_seis = pd.read_csv(seis_file, parse_dates=["Date"])
df_so2  = pd.read_csv(so2_file, parse_dates=["Date"])

#  Merge datasets based on daily date 
df = (
    df_frp.merge(df_seis, on="Date", how="inner")
          .merge(df_so2,  on="Date", how="inner")
)

# Standardise names for consistency
# -FRP - daily MW detected from MODIS Fire Product
# -NosQuakes - daily numnber of earthquakes in vicinity
# -MeanMagQuakes - mean magnitude of the earthquakes
# -SO2 detected from NASA’s Ozone Monitoring Instrument, in Dobson Units, column SO2

df = df.rename(columns={
    "frp": "FRP",
    "NosQuakes": "NosQuakes",
    "MeanMagQuakes": "MeanMag",
    "SO2": "SO2"
})

#  Makes tomorrow's FRP the target based on today's data
df["FRP_tomorrow"] = df["FRP"].shift(-1)

# Drop last row (no target available)
df = df.dropna()

#  x - the input variables of today, y - the desired output; selects these columns, removes the header, and converts to NumnPy array, as needed by the ML tool, RandomForestRegressor
X = df[["NosQuakes", "MeanMag", "SO2", "FRP"]].values
y = df["FRP_tomorrow"].values

# prints first 5 rows and labels of data
print(df[["NosQuakes", "MeanMag", "SO2", "FRP", "FRP_tomorrow"]].head())


#  take 80% of the data for training (current day, x, and next, y)
split = int(0.8 * len(X))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Run random forest model; see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# Grow x200 decision trees
model = RandomForestRegressor(n_estimators=200, random_state=42)
# fits to the model
model.fit(X_train, y_train)
# predicts y from x
y_pred = model.predict(X_test)

#  Compares predicted and actual 
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))

# How important was each feature
importances = model.feature_importances_
labels = ["NosQuakes", "MeanMag", "SO2", "FRP"]

# Create figure and axis explicitly
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(labels, importances)
ax.set_title("Feature Importance for Predicting Next-Day FRP")
fig.tight_layout()

# --- Save to results folder ---
import os
os.makedirs("results", exist_ok=True)
plot_path = os.path.join("results", "feature_importance.png")
fig.savefig(plot_path, dpi=150)   # <-- save using the fig object
print(f"Feature importance plot saved to: {plot_path}")

# --- Show on screen ---
plt.show(block=True)


