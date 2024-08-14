import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import rmse
from darts.dataprocessing.transformers import Scaler
from get_weather import weather

# df = df[df['Direction'] == 'CC']
# # df = df[df['Route'] == 142]
# # #df['time_to_stop'] = df.groupby('Date')['Arrival'].apply(lambda x: x - df['Departure'].shift(1)).dt.total_seconds()
# # df['time_to_stop'] = df.groupby('Date').apply(lambda group: group['Arrival'] - group['Departure'].shift(1)).reset_index(level=0, drop=True).dt.total_seconds()
# # print(df['time_to_stop'].max())
# # print(df['time_to_stop'].min())
# # print(len(df[df['time_to_stop'] < 0]))
# df = df[df['Trip'] == 1408]
# df.to_excel('trip_1408.xlsx')
# df['time'] = (df['Departure'] - df['Departure'].shift(1)).dt.total_seconds()
# print(df[df['time'] < 0])

df = pd.read_excel('historical.xlsx', sheet_name='Spring 2023')
# df_2 = pd.read_excel('historical.xlsx', sheet_name='Fall 2023')
# for df in [df_1, df_2]
# df = df.sort_values(by=['Date', 'Arrival'])

#Look at dwell time only for now

df = df[['Date', 'Day of Week', 'Dwell (s)']]
grouped = df.groupby('Date').agg({'Dwell (s)': 'mean', 'Day of Week': 'first'}).reset_index()
grouped['Date'] = pd.to_datetime(grouped['Date'])
grouped.set_index('Date', inplace=True)
full_date_range = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq='B')

df_reindexed = grouped.reindex(full_date_range)
df_reindexed['Day of Week'] = df_reindexed.index.day_name()

df_reindexed['Dwell (s)'] = df_reindexed['Dwell (s)'].interpolate(method='linear')
df_reindexed.reset_index(inplace=True)
df_reindexed = df_reindexed.rename(columns={'index':'Date'})
df_reindexed['day'] = df_reindexed['Day of Week'].map({'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5})


weather_df = weather(df_reindexed['Date'].min().strftime('%Y-%m-%d'), df_reindexed['Date'].max().strftime('%Y-%m-%d'))
weather_df['date'] = weather_df['date'].dt.tz_convert(None)
df_reindexed = pd.merge(df_reindexed, weather_df, how='left', left_on='Date', right_on='date')
df_reindexed = df_reindexed.drop(['date', 'day_of_week'], axis=1)
df_reindexed['month'] = df_reindexed['month'].map({'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                                                   'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
print(df_reindexed.columns)
test_length = 15
train_df = df_reindexed.iloc[:-test_length]
test_df = df_reindexed.iloc[-test_length:]

train_series = TimeSeries.from_dataframe(train_df, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)
test_series = TimeSeries.from_dataframe(test_df, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)

covariate = TimeSeries.from_dataframe(df_reindexed, 'Date', 'day', freq='B')
scaler = Scaler()
covariate = scaler.fit_transform(covariate)

covariate1 = TimeSeries.from_dataframe(df_reindexed, 'Date', 'apparent_temperature', freq='B')
scaler = Scaler()
covariate1 = scaler.fit_transform(covariate1)
covariate = covariate.stack(covariate1)

covariate2 = TimeSeries.from_dataframe(df_reindexed, 'Date', 'sunshine', freq='B')
scaler = Scaler()
covariate2 = scaler.fit_transform(covariate2)
covariate = covariate.stack(covariate2)

covariate3 = TimeSeries.from_dataframe(df_reindexed, 'Date', 'precipitation', freq='B')
scaler = Scaler()
covariate3 = scaler.fit_transform(covariate3)
covariate = covariate.stack(covariate3)

transformer = Scaler()
transformer.fit(train_series)
train_series = transformer.transform(train_series)
test_series = transformer.transform(test_series)

model = ARIMA()
model.fit(train_series, future_covariates=covariate)
forecast = model.predict(len(test_df), future_covariates=covariate)
forecast = forecast.slice_intersect(test_series)
error = rmse(test_series, forecast)
print(error)
plt.plot(df_reindexed['Date'][:-test_length], train_series.values(), color='black', label='train')
plt.plot(df_reindexed['Date'][-test_length:], test_series.values(), color='black', label='test')
plt.plot(df_reindexed['Date'][-test_length:], forecast.values(), color='blue', label='forecast')
plt.legend()
plt.show()