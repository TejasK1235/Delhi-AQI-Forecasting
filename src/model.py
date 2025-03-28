import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pickle

df = pd.read_csv('data/processed/delhi_aqi_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df.fillna(method='ffill',inplace=True)
df.set_index('date',inplace=True)
df = df.asfreq('h')

df_resampled = df['AQI'].resample('W').mean()

adf_test = adfuller(df['AQI'])

# ARIMA(1,0,0)
model_100 = ARIMA(df['AQI'], order=(1, 0, 0))
result_100 = model_100.fit()

# ARIMA(1,0,1)
model_101 = ARIMA(df['AQI'], order=(1, 0, 1))
result_101 = model_101.fit()

# ARIMA Logged (1,0,1)
df['AQI_log'] = np.log(df['AQI'])
model_log = ARIMA(df['AQI_log'], order=(1, 0, 1))
result_log = model_log.fit()

with open("models/arima_101.pkl", "wb") as f:
    pickle.dump(result_101, f)

with open("models/arima_log_101.pkl", "wb") as f:
    pickle.dump(result_log, f)
