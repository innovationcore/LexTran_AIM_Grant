import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import rmse
from darts.dataprocessing.transformers import Scaler
from get_weather import weather
from scipy.stats import ttest_ind


def forecast_dwell(df_1, df_2, df_3):
    new_dfs = []
    for df in [df_1, df_2, df_3]:
        df = df.sort_values(by=['Date', 'Arrival'])

        # Look at dwell time only for now

        df = df[['Date', 'Day of Week', 'Dwell (s)']]
        grouped = df.groupby('Date').agg({'Dwell (s)': 'mean', 'Day of Week': 'first'}).reset_index()
        grouped['Date'] = pd.to_datetime(grouped['Date'])
        grouped.set_index('Date', inplace=True)
        full_date_range = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq='B')

        df_reindexed = grouped.reindex(full_date_range)
        df_reindexed['Day of Week'] = df_reindexed.index.day_name()

        df_reindexed['Dwell (s)'] = df_reindexed['Dwell (s)'].interpolate(method='linear')
        df_reindexed.reset_index(inplace=True)
        df_reindexed = df_reindexed.rename(columns={'index': 'Date'})
        df_reindexed['day'] = df_reindexed['Day of Week'].map(
            {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5})

        weather_df = weather(df_reindexed['Date'].min().strftime('%Y-%m-%d'),
                             df_reindexed['Date'].max().strftime('%Y-%m-%d'))
        weather_df['date'] = weather_df['date'].dt.tz_convert(None)
        df_reindexed = pd.merge(df_reindexed, weather_df, how='left', left_on='Date', right_on='date')
        df_reindexed = df_reindexed.drop(['date', 'day_of_week'], axis=1)
        df_reindexed['month'] = df_reindexed['month'].map(
            {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
        new_dfs.append(df_reindexed)
    spring = new_dfs[0]
    fall = new_dfs[1]
    pilot = new_dfs[2]
    test_length = 15
    train_pilot = pilot.iloc[:-test_length]
    test_pilot = pilot.iloc[-test_length:]

    spring_series = TimeSeries.from_dataframe(spring, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)
    fall_series = TimeSeries.from_dataframe(fall, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)
    pilot_train_series = TimeSeries.from_dataframe(train_pilot, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)
    pilot_test_series = TimeSeries.from_dataframe(test_pilot, 'Date', 'Dwell (s)', freq='B', fill_missing_dates=False)

    transformer = Scaler()
    transformer.fit([spring_series, fall_series, pilot_train_series])
    spring_series = transformer.transform(spring_series)
    fall_series = transformer.transform(fall_series)
    pilot_train_series = transformer.transform(pilot_train_series)
    pilot_test_series = transformer.transform(pilot_test_series)

    covariates = []
    for df in [spring, fall, pilot]:
        covariate = TimeSeries.from_dataframe(df, 'Date', 'day', freq='B')
        scaler = Scaler()
        covariate = scaler.fit_transform(covariate)

        covariate1 = TimeSeries.from_dataframe(df, 'Date', 'apparent_temperature', freq='B')
        scaler = Scaler()
        covariate1 = scaler.fit_transform(covariate1)
        covariate = covariate.stack(covariate1)

        covariate2 = TimeSeries.from_dataframe(df, 'Date', 'sunshine', freq='B')
        scaler = Scaler()
        covariate2 = scaler.fit_transform(covariate2)
        covariate = covariate.stack(covariate2)

        covariate3 = TimeSeries.from_dataframe(df, 'Date', 'precipitation', freq='B')
        scaler = Scaler()
        covariate3 = scaler.fit_transform(covariate3)
        covariate = covariate.stack(covariate3)
        covariates.append(covariate)
    spring_covariates = covariates[0]
    fall_covariates = covariates[1]
    pilot_covariates = covariates[2]

    model = ARIMA(q=0, d=1, p=8)
    model.fit(spring_series, future_covariates=spring_covariates)
    model.fit(fall_series, future_covariates=fall_covariates)
    model.fit(pilot_train_series, future_covariates=pilot_covariates)
    forecast = model.predict(len(test_pilot), future_covariates=pilot_covariates)
    forecast = forecast.slice_intersect(pilot_test_series)
    error = rmse(pilot_test_series, forecast)
    print(error)
    # plt.plot(spring['Date'], spring_series.values(), color='black', label='spring')
    plt.plot(pilot['Date'][:-test_length], pilot_train_series.values(), color='black', label='fall')
    plt.plot(pilot['Date'][-test_length:], pilot_test_series.values(), color='black', label='test')
    plt.plot(pilot['Date'][-test_length:], forecast.values(), color='blue', label='forecast')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title('Forecasting Dwell Time by Day- RMSE: {:.4f}'.format(error))
    plt.show()

    plt.plot(spring['Date'], spring['Dwell (s)'], label='Spring 2023')
    plt.plot(fall['Date'], fall['Dwell (s)'], label='Fall 2023')
    plt.plot(pilot['Date'], pilot['Dwell (s)'], label='Spring 2024')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title('Dwell Time by Semester')
    plt.show()

def forecast_travel(df_1, df_2, df_3):
    new_dfs = []
    for df in [df_1, df_2, df_3]:
        df = df.dropna()
        df = df.sort_values(by=['Date', 'Arrival'])

        # Look at dwell time only for now

        df = df[['Date', 'Day of Week', 'Time_to_Stop']]
        grouped = df.groupby('Date').agg({'Time_to_Stop': 'mean', 'Day of Week': 'first'}).reset_index()
        grouped['Date'] = pd.to_datetime(grouped['Date'])
        grouped.set_index('Date', inplace=True)
        full_date_range = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq='B')

        df_reindexed = grouped.reindex(full_date_range)
        df_reindexed['Day of Week'] = df_reindexed.index.day_name()

        df_reindexed['Time_to_Stop'] = df_reindexed['Time_to_Stop'].interpolate(method='linear')
        df_reindexed.reset_index(inplace=True)
        df_reindexed = df_reindexed.rename(columns={'index': 'Date'})
        df_reindexed['day'] = df_reindexed['Day of Week'].map(
            {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5})

        weather_df = weather(df_reindexed['Date'].min().strftime('%Y-%m-%d'),
                             df_reindexed['Date'].max().strftime('%Y-%m-%d'))
        weather_df['date'] = weather_df['date'].dt.tz_convert(None)
        df_reindexed = pd.merge(df_reindexed, weather_df, how='left', left_on='Date', right_on='date')
        df_reindexed = df_reindexed.drop(['date', 'day_of_week'], axis=1)
        df_reindexed['month'] = df_reindexed['month'].map(
            {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})
        new_dfs.append(df_reindexed)
    spring = new_dfs[0]
    fall = new_dfs[1]
    pilot = new_dfs[2]
    test_length = 15
    train_pilot = pilot.iloc[:-test_length]
    test_pilot = pilot.iloc[-test_length:]

    spring_series = TimeSeries.from_dataframe(spring, 'Date', 'Time_to_Stop', freq='B', fill_missing_dates=False)
    fall_series = TimeSeries.from_dataframe(fall, 'Date', 'Time_to_Stop', freq='B', fill_missing_dates=False)
    pilot_train_series = TimeSeries.from_dataframe(train_pilot, 'Date', 'Time_to_Stop', freq='B', fill_missing_dates=False)
    pilot_test_series = TimeSeries.from_dataframe(test_pilot, 'Date', 'Time_to_Stop', freq='B', fill_missing_dates=False)

    transformer = Scaler()
    transformer.fit([spring_series, fall_series, pilot_train_series])
    spring_series = transformer.transform(spring_series)
    fall_series = transformer.transform(fall_series)
    pilot_train_series = transformer.transform(pilot_train_series)
    pilot_test_series = transformer.transform(pilot_test_series)

    covariates = []
    for df in [spring, fall, pilot]:
        covariate = TimeSeries.from_dataframe(df, 'Date', 'day', freq='B')
        scaler = Scaler()
        covariate = scaler.fit_transform(covariate)

        covariate1 = TimeSeries.from_dataframe(df, 'Date', 'apparent_temperature', freq='B')
        scaler = Scaler()
        covariate1 = scaler.fit_transform(covariate1)
        covariate = covariate.stack(covariate1)

        covariate2 = TimeSeries.from_dataframe(df, 'Date', 'sunshine', freq='B')
        scaler = Scaler()
        covariate2 = scaler.fit_transform(covariate2)
        covariate = covariate.stack(covariate2)

        covariate3 = TimeSeries.from_dataframe(df, 'Date', 'precipitation', freq='B')
        scaler = Scaler()
        covariate3 = scaler.fit_transform(covariate3)
        covariate = covariate.stack(covariate3)
        covariates.append(covariate)
    spring_covariates = covariates[0]
    fall_covariates = covariates[1]
    pilot_covariates = covariates[2]

    model = ARIMA(q=0, d=1, p=10)
    model.fit(spring_series, future_covariates=spring_covariates)
    model.fit(fall_series, future_covariates=fall_covariates)
    model.fit(pilot_train_series, future_covariates=pilot_covariates)
    forecast = model.predict(len(test_pilot), future_covariates=pilot_covariates)
    forecast = forecast.slice_intersect(pilot_test_series)
    error = rmse(pilot_test_series, forecast)
    print(error)
    # plt.plot(spring['Date'], spring_series.values(), color='black', label='spring')
    plt.plot(pilot['Date'][:-test_length], pilot_train_series.values(), color='black', label='fall')
    plt.plot(pilot['Date'][-test_length:], pilot_test_series.values(), color='black', label='test')
    plt.plot(pilot['Date'][-test_length:], forecast.values(), color='blue', label='forecast')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title('Forecasting Travel Time by Day- RMSE: {:.4f}'.format(error))
    plt.show()

    plt.plot(spring['Date'], spring['Time_to_Stop'], label='Spring 2023')
    plt.plot(fall['Date'], fall['Time_to_Stop'], label='Fall 2023')
    plt.plot(pilot['Date'], pilot['Time_to_Stop'], label='Spring 2024')
    plt.legend()
    plt.xticks(rotation=30)
    plt.title('Travel Time by Semester')
    plt.show()

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

df_1 = pd.read_excel('historical.xlsx', sheet_name='Fall 2023') #Actually spring
df_2 = pd.read_excel('historical.xlsx', sheet_name='Spring 2023') #Actually fall
df_3 = pd.read_excel('pilot_grant.xlsx', sheet_name='Sheet1') #Spring 2024
# print(ttest_ind(df_2['Dwell (s)'], df_3['Dwell (s)'], alternative='two-sided'))
# print(df_2['Dwell (s)'].mean())
# print(df_3['Dwell (s)'].mean())
# exit()
#df = df_2.groupby(['Date', 'Route', 'Direction', 'Trip']).apply(lambda group: (group['Arrival'] - group['Departure'].shift(1)).dt.total_seconds())

df_spring = df_1.groupby(['Date', 'Route', 'Direction', 'Trip']).apply(
    lambda group: group.sort_values('Arrival').assign(
        Time_to_Stop=(group['Arrival'] - group['Departure'].shift(1)).dt.total_seconds()
    )
).drop(['Date', 'Route', 'Direction', 'Trip'], axis=1).reset_index()
df_spring = df_spring[df_spring['Time_to_Stop'] >= 0]

df_fall = df_2.groupby(['Date', 'Route', 'Direction', 'Trip']).apply(
    lambda group: group.sort_values('Arrival').assign(
        Time_to_Stop=(group['Arrival'] - group['Departure'].shift(1)).dt.total_seconds()
    )
).drop(['Date', 'Route', 'Direction', 'Trip'], axis=1).reset_index()
df_fall = df_fall[df_fall['Time_to_Stop'] >= 0]

df_pilot = df_3.groupby(['Date', 'Route', 'Direction', 'Trip']).apply(
    lambda group: group.sort_values('Arrival').assign(
        Time_to_Stop=(group['Arrival'] - group['Departure'].shift(1)).dt.total_seconds()
    )
).drop(['Date', 'Route', 'Direction', 'Trip'], axis=1).reset_index()
df_pilot = df_pilot[df_pilot['Time_to_Stop'] >= 0]

forecast_travel(df_spring, df_fall, df_pilot)
#forecast_dwell(df_spring, df_fall, df_pilot)
# print(ttest_ind(df_fall['Time_to_Stop'], df_pilot['Time_to_Stop'], alternative='two-sided'))
# print(df_fall['Time_to_Stop'].mean())
# print(df_pilot['Time_to_Stop'].mean())
