# Volcanic Data AI Application

This repository contains Python scripts and datasets for analysing volcanic precursor signals and filling gaps in volcanic observation data.  
It combines machine learning (Random Forest) and deep learning (Neural Networks) approaches using daily Fire Radiative Power (FRP), SO₂, and seismic data from Kilauea volcano (2004–2025).
 
You will run 3 programs with different purposes:

1. `scripts/PredNextDay_FRP.py` - To predict the next day's FRP from today's FRP, SO₂, and seismic data
- Loads daily FRP, SO₂, and seismic datasets.
- Trains a Random Forest Regressor to predict tomorrow’s FRP from today’s values.
- Prints model accuracy (RMSE and R²).
- Produces a feature importance plot showing which variables matter most.
Question: which, if any, dataset is most important in determining future volcanic behaviour?

2. `scripts/Precursors_All.py` - To find common precursors in the data before the actual dates of eruptions at Kilauea
- Labels eruption “precursor windows” (30 days before historical eruptions at Kilauea).
- Trains a Neural Network Autoencoder on daily FRP, seismicity, and SO₂.
- Computes reconstruction error to highlight unusual behaviour.
- Produces summary table comparing quiet vs precursor periods.
- Outputs plot of anomaly scores with eruption windows highlighted.
  Question: are there any datasets which provide potentially useful precursory information?

3. `scripts/CloudFiller.py` - To fill in gaps in the datsets and produce 'complete' datasets to feed back into 1 & 2
- Examines raw FRP data with gaps (e.g. cloudy days) and uses Random Forest FRP, SO₂ and seismic features to fill in missing values.
- Outputs a new CSV with observed + filled values.
- Produces a time-series plot highlighting the filled data points.
- Re-run programs 1 and 2 using outputs of this program.
Question: Did gap filling improve the performance of your prediction / precursory models?

--------------

Raw datasets sources:
- FRP: MODIS day and night, from NASA FIRMS for region around Kilauea, 2004-2025, averaged for daily values
- Seismic data: All magnitude from https://earthquake.usgs.gov/fdsnws/event/1/query, with output as mean mag and number per day
- SO₂: from NASA Giovanni, https://giovanni.gsfc.nasa.gov/, O2 Column Amount (OMSO2e v003), region (Kilauea)

----------

volcanic-data-ai-application/
│
├── data/ # Input datasets (CSV)
│ ├── Kilauea_Daily_FRP.csv # Daily Fire Radiative Power (MW, MODIS product)
│ ├── Kilauea_Daily_Seismic.csv # Daily number and mean magnitude of earthquakes
│ ├── Kilauea_Daily_SO2.csv # Daily SO₂ column amounts (Dobson Units, OMI satellite)
│
├── scripts/ # Analysis scripts
│ ├── PredNextDay_FRP.py # Predicts tomorrow’s FRP from today’s FRP, SO₂, and seismicity
│ ├── CloudFiller.py # Fills missing FRP values (e.g., cloudy days) using Random Forest
│ ├── Precursors_All.py # Detects eruption precursor anomalies using a neural network autoencoder
│
├── results/ # Outputs (plots, CSVs will be written here)
│
├── requirements.txt # Python dependencies
└── README.md # This file

## How to Run

1. Clone this repo and enter the folder:
   git clone https://github.com/drblackett/volcanic-data-ai-application.git
   cd volcanic-data-ai-application

2. pip install -r requirements.txt

3. Run any program e.g. python scripts/PredNextDay_FRP.py
