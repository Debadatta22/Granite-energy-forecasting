# Granite-energy-forecasting
Time series forecasting of Spain’s hourly electricity demand using IBM Granite TinyTimeMixer (TTM) models with Hugging Face datasets and visualization.

___________________

______________________

📌 Energy Demand Forecasting with IBM Granite Time Series (TTM)
🔹 Introduction

Forecasting in time series analysis enables data scientists to identify historical patterns and generate predictions about the future. In this project, we leverage IBM’s Granite Time Series Foundation Models (TSFM) and the TinyTimeMixers (TTM) — compact pre-trained models for multivariate time-series forecasting.

TinyTimeMixers (TTM) contain fewer than 1 million parameters yet outperform traditional models with billions of parameters in zero-shot and few-shot forecasting tasks.

They can be fine-tuned easily for multivariate forecasts while being lightweight and efficient.

The granite-tsfm library provides all necessary utilities for working with TSFM.

The focus of this lab is to build an Energy Demand Forecasting pipeline using Spain’s hourly electricity consumption dataset, enabling us to understand consumption patterns and predict future demand.

🔹 Problem Statement

The challenge is to predict future electricity demand based on historical data. Since energy consumption shows strong cyclical and seasonal patterns (daily/weekly usage trends), forecasting models are essential for grid management, load balancing, and resource allocation.

🔹 Approach & Solution

We adopt the following approach:

Install Dependencies – Setup the environment with the granite-tsfm library.

Import Libraries – Load essential libraries for data handling, modeling, and visualization.

Load Dataset – Use Spain’s energy consumption dataset hosted on Hugging Face.

Data Preparation – Clean and preprocess data, selecting a fixed context length.

Visualization of Historical Data – Plot past demand trends.

Model Setup – Load IBM’s pre-trained TTM model.

Forecasting Pipeline – Configure the pipeline and run predictions.

Visualize Forecast – Compare historical vs. predicted demand on a time-series graph.

🔹 Step-by-Step Breakdown
1. Install the TSFM Library
! pip install "granite-tsfm[notebooks]==0.2.23"


Motive: Set up the environment.

Role: Installs the IBM Granite TSFM package with notebook dependencies.

Logic: Uses pip to fetch version 0.2.23 ensuring compatibility and stability.

2. Import Packages
import matplotlib.pyplot as plt
import pandas as pd
import torch

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)
from tsfm_public.toolkit.visualization import plot_predictions


Motive: Make necessary tools available.

Role:

matplotlib → Data visualization.

pandas → Data manipulation.

torch → GPU/CPU execution.

TinyTimeMixerForPrediction → Pre-trained TTM model.

TimeSeriesForecastingPipeline → Configures forecasting workflow.

plot_predictions → Visualizes results.

3. Dataset Path
DATA_FILE_PATH = "hf://datasets/vitaliy-sharandin/energy-consumption-hourly-spain/energy_dataset.csv"


Motive: Define data source.

Role: Provides Hugging Face dataset path.

Logic: Directs model to Spain’s hourly energy demand dataset.

4. Define Time & Target Variables
timestamp_column = "time"
target_columns = ["total load actual"]
context_length = 512


Motive: Identify inputs & outputs.

Role:

timestamp_column → Time reference.

target_columns → Target variable to forecast (total load actual).

context_length → Number of past data points used by the model.

5. Load & Prepare Data
input_df = pd.read_csv(DATA_FILE_PATH, parse_dates=[timestamp_column])
input_df = input_df.ffill()
input_df = input_df.iloc[-context_length: ,]
input_df.tail()


Motive: Clean dataset for forecasting.

Steps:

Load dataset with timestamps parsed as dates.

Handle missing values using forward fill.

Select last 512 observations (model context).

Display preview with .tail().

6. Plot Target Series
fig, axs = plt.subplots(len(target_columns), 1, figsize=(10, 2 * len(target_columns)), squeeze=False)
for ax, target_column in zip(axs, target_columns):
    ax[0].plot(input_df[timestamp_column], input_df[target_column])


Motive: Inspect historical demand.

Role: Generates line plot(s) of total load actual against time.

7. Load Pre-trained TinyTimeMixer Model
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",
    num_input_channels=len(target_columns),
)


Motive: Bring IBM’s pre-trained model into workflow.

Role: Loads Granite TTM-R2 model from Hugging Face.

Logic: num_input_channels matches number of target columns (here, 1).

8. Forecasting Pipeline Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    timestamp_column=timestamp_column,
    id_columns=[],
    target_columns=target_columns,
    explode_forecasts=False,
    freq="h",
    device=device,
)

zeroshot_forecast = pipeline(input_df)
zeroshot_forecast.tail()


Motive: Create structured forecasting pipeline.

Steps:

Detect execution device (CPU/GPU).

Configure pipeline with:

Model

Frequency ("h" = hourly)

Target column(s)

Forecasting settings (explode_forecasts=False)

Generate predictions (zeroshot_forecast).

9. Plot Predictions vs Historical Data
plot_predictions(
    input_df=input_df,
    predictions_df=zeroshot_forecast,
    freq="h",
    timestamp_column=timestamp_column,
    channel=target_column,
    indices=[-1],
    num_plots=1,
)


Motive: Visual comparison of real vs predicted values.

Role: Overlay historical demand with forecasted values.

Insight: Model predicts continuation of cyclical patterns with potential upward trend.

🔹 Key Insights

Granite TTMs are efficient and lightweight, making them suitable for quick deployment in real-world forecasting.

The pipeline supports zero-shot predictions, requiring no fine-tuning.

This workflow demonstrates end-to-end forecasting: from data preparation → modeling → visualization.

🔹 Skills You’ll Learn

Time series forecasting using pre-trained foundation models.

Data preprocessing and handling missing values.

Model deployment with Hugging Face pre-trained assets.

Visualization of time series and forecasts.
