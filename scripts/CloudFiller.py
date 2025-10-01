# Examines raw FRP data time series, finds missing or invalid values, and uses machine learning (Random Forest) with yesterday’s FRP, quakes
# mean magnitude and SO₂ to predict and fill those gaps. Outputs both a CSV and a plot so you can check the results.
# To run in Spyder for example, be in upper directory (volcanic-data-ai-application/) and run via: !python scripts\CloudFiller.py

# sklearn - machine learning library; ensamble - module which contains the RandomForestRegressor class
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data from local 'data' folder 

data_dir = "data"

frp_file  = os.path.join(data_dir, "Kilauea_Daily_FRP.csv")
seis_file = os.path.join(data_dir, "Kilauea_Daily_Seismic.csv")
so2_file  = os.path.join(data_dir, "Kilauea_Daily_SO2.csv")

df_frp  = pd.read_csv(frp_file, parse_dates=["Date"])
df_seis = pd.read_csv(seis_file, parse_dates=["Date"])
df_so2  = pd.read_csv(so2_file, parse_dates=["Date"])

# Merge datasets on Date
df = (df_frp.merge(df_seis, on="Date", how="inner")
            .merge(df_so2,  on="Date", how="inner"))

# Standardise column names for consistency
df = df.rename(columns={
    "frp": "FRP",
    "MeanMagQuakes": "MeanMag"
}).set_index("Date").sort_index()

# Configure random forest for gap fill (cf. PredNextDay.py), with 200 decision trees, 10 branches, 42 seeds
base_features = [c for c in ["Cloud","NosQuakes","MeanMag","SO2"] if c in df.columns]
N_EST, MAX_DEPTH, RS = 200, 10, 42


def fit_and_fill_with_lag1(target_col="FRP"):
    y_obs = df[target_col].astype(float)

    # define gaps as missing, do be filled
    is_gap = y_obs.isna() | (y_obs <= 0)

    # set up to use today’s seismicity, SO₂, and cloud cover plus yesterday’s FRP to help predict today’s FRP
    lag1 = y_obs.shift(1)
    X_all = pd.concat([df[base_features], lag1.rename("lag1")], axis=1)

    # Training based on no missing features
    train_mask = (~is_gap) & X_all.notna().all(axis=1)
    X_train = X_all.loc[train_mask]
    y_train = y_obs.loc[train_mask]

    # FRP data has lots of small values, a few extreme ones, so this helps the Random Forest focus on patterns instead of extremes.
    y_train_log = np.log1p(y_train.clip(lower=0))

    #makes & trains forest
    rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH,
                               random_state=RS, n_jobs=-1)
    rf.fit(X_train, y_train_log)

    # predict for all days
    X_pred = X_all.copy()
    X_pred["lag1"] = X_pred["lag1"].fillna(0.0)
    have_feats = X_pred.notna().all(axis=1)
    y_hat_log = pd.Series(np.nan, index=df.index)
    y_hat_log.loc[have_feats] = rf.predict(X_pred.loc[have_feats])
    y_hat = np.expm1(y_hat_log.clip(lower=0))

    # combines observed + filled
    filled = y_obs.copy()
    to_fill = is_gap & have_feats
    filled.loc[to_fill] = y_hat.loc[to_fill]

    return filled, to_fill

# Run function
filled_frp, gap_points = fit_and_fill_with_lag1("FRP")

# Save results
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "FRP_gapfilled.csv")
out_df = df.copy()
out_df["FRP_filled"] = filled_frp
out_df["FRP_gap_filled"] = gap_points
out_df.to_csv(out_path)
print(f"Gap-filled FRP saved to: {out_path}")

# Plot observed vs filled
fig, ax = plt.subplots(figsize=(12,5))

# Plot observed FRP (with gaps)
ax.plot(df.index, df["FRP"], "k.", markersize=2, label="Observed FRP", zorder=1)

# Plot gap-filled series
ax.plot(df.index, filled_frp, "b-", linewidth=1, label="Gap-filled FRP", zorder=2)

# Overlay filled points in red (smaller, on top)
ax.scatter(df.index[gap_points], filled_frp[gap_points],
           color="red", s=1, label="Filled points", zorder=3)   # << smaller size

ax.set_title("FRP Gap-Filled")
ax.legend()
plt.tight_layout()

# Save + show
plot_path = os.path.join("results", "FRP_gapfilled.png")
plt.savefig(plot_path, dpi=150)
print(f"Gap-fill plot saved to: {plot_path}")

plt.show()
