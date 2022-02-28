import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

df1 = pd.read_csv('Time_series_monthly.csv')
df1.index = df1['datetime_utc']
df1.drop('datetime_utc',axis=1,inplace=True)
df1.index = pd.DatetimeIndex(df1.index).to_period('M')

pred_date = df1.copy()
pred_date.set_index(df1.index,inplace=True)

for i in range(0,50):
    pred_date = pred_date.append({' _tempm':np.NaN},ignore_index=True)

model = SARIMAX(pred_date[' _tempm'],order = (1,0,1),seasonal_order=(1,0,1,12))
results = model.fit()
pred_date['forecast']=results.predict(start=235,end=290,dynamic=True)

idx = df1.index
months = []
for i in range(0,290):
    months.append(idx[0]+i)

pred_date['Months'] = months
pred_date.set_index('Months',inplace=True)
pred_date[[' _tempm','forecast']].plot(figsize=(10,5))
pred_date.to_csv('Monthly_Predictions.csv')

