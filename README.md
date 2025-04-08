# Delhi AQI Forecasting Using Time Series Analysis

This project forecasts Delhi's Air Quality Index (AQI) using ARIMA time series models based on historical pollution data. The goal is to provide insights into air quality trends and make short-term forecasts using statistically sound methods.

## Dataset

The dataset is sourced from the official Delhi AQI monitoring website [aqi.in](https://www.aqi.in). While the original dataset contained multiple features such as PM10, PM2.5, NO2, SO2, etc., the data was cleaned and processed using the official AQI calculation formula to obtain a final dataset with two columns: `timestamp` and `AQI`.

- **Time Range:** From the beginning of 2021 to end of 2023
- **Frequency:** Hourly

## Model

This project uses the ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting. Different configurations were explored to model the AQI data effectively.

## Visualizations

The `plots` folder contains various visualizations, including:
- ACF and PACF plots
- Forecasted AQI values vs actual values (up to date)

## Getting Started

To run the notebook, clone the repository and install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
