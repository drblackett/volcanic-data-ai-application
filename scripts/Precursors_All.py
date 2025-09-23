import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Load data from local 'data' folder ---
data_dir = "data"

frp_file  = os.path.join(data_dir, "Kilauea_Daily_FRP.csv")
seis_file = os.path.join(data_dir, "Kilauea_Daily_Seismic.csv")
so2_file  = os.path.join(data_dir, "Kilauea_Daily_SO2.csv")

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
 
# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# --- Define eruption dates ---
ERUPTIONS = pd.to_datetime([
    "2007-06-17","2007-07-21","2008-03-19","2011-03-05","2014-06-27",
    "2016-05-24","2018-05-03","2020-12-20","2021-09-29","2023-01-05",
    "2023-06-07","2023-09-10","2024-06-03","2024-09-15","2024-12-23"
])
lookback = 30  # days before eruption

# --- Label eruption windows ---
df["eruption_window"] = 0
for e in ERUPTIONS:
    mask = (df["Date"] >= e - pd.Timedelta(days=lookback)) & (df["Date"] < e)
    df.loc[mask, "eruption_window"] = 1

# --- Features ---
features = ["FRP", "NosQuakes", "MeanMag", "SO2"]
X = df[features].fillna(0).values

# --- Autoencoder model ---
model = Sequential([
    Dense(8, activation="relu", input_shape=(X.shape[1],)),
    Dense(4, activation="relu"),
    Dense(8, activation="relu"),
    Dense(X.shape[1], activation="linear")
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, X, epochs=20, batch_size=32, verbose=0)

# --- Reconstruction error per feature ---
X_pred = model.predict(X)
errors = (X - X_pred)**2

df["err_frp"]     = errors[:,0]
df["err_quakes"]  = errors[:,1]
df["err_meanmag"] = errors[:,2]
df["err_so2"]     = errors[:,3]

# --- Summary table (quiet vs precursor) ---
summary = df.groupby("eruption_window")[["err_frp","err_quakes","err_meanmag","err_so2"]].mean()
summary = summary.rename(index={0:"Quiet", 1:"Precursor"})
print("\nAverage reconstruction error by feature:")
print(summary)

# --- Plot per-feature anomaly scores ---
plt.figure(figsize=(10,6))
plt.plot(df["Date"], df["err_frp"],     label="FRP error")
plt.plot(df["Date"], df["err_quakes"],  label="NosQuakes error")
plt.plot(df["Date"], df["err_so2"],     label="SO₂ error")

# Shade eruption windows
for e in ERUPTIONS:
    plt.axvspan(e - pd.Timedelta(days=lookback), e, color="red", alpha=0.2)

plt.title("Per-feature anomaly scores vs eruptions")
plt.xlabel("Date")
plt.ylabel("Reconstruction error")
plt.legend()
plt.tight_layout()

# Save plot
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "per_feature_errors.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")

plt.show()
