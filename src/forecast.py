import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("data/processed/delhi_aqi_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

with open("models/arima_101.pkl", "rb") as f:
    model = pickle.load(f)

# Forecast for the next 30 days
forecast_steps = 30
forecast = model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(df.index[-500:], df['AQI'].iloc[-500:], label='Actual AQI', color='blue')
plt.plot(forecast_mean.index, forecast_mean, label='Forecasted AQI', color='red')
plt.title("AQI Forecast")
plt.legend()

plt.savefig("plots/aqi_forecast.png")