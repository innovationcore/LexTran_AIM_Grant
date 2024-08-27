import pandas as pd

def categorize_headway(value):
    if value < 4:
        return 'Far Below Ideal'
    if value < 7:
        return 'Below Ideal'
    elif 7 <= value <= 10:
        return 'Ideal'
    elif value < 15:
        return 'Above Ideal'
    else:
        return 'Far Above Ideal'

og_df = pd.read_excel('pilot_grant.xlsx', sheet_name='Sheet1')
new_df = pd.DataFrame()
for date in og_df['Date'].unique():
    df = og_df[og_df['Date'] == date]
    for direction in ['CC', 'CW']:
        data = df[df['Direction'] == direction]
        data = data[['Stop_Name', 'Trip', 'Departure']]
        data = data.sort_values(by=['Stop_Name', 'Trip', 'Departure'])
        data['time_diff'] = data.groupby(['Stop_Name', 'Trip'])['Departure'].diff()

        # Identify rows to drop
        mask = (data['time_diff'] <= pd.Timedelta(minutes=1))

        # Drop the rows that satisfy the condition (i.e., the second row in the consecutive pair)
        df_filtered = data[~mask].drop(columns='time_diff')
        df_filtered = df_filtered.sort_values(by=['Stop_Name', 'Departure'])
        df_filtered['headway'] = (df_filtered.groupby('Stop_Name')['Departure'].diff()).dt.total_seconds() / 60
        df_filtered['Date'] = date
        new_df = pd.concat([new_df, df_filtered])

new_df = new_df[~new_df['headway'].isna()]
new_df['headway_adherence'] = new_df['headway'].apply(categorize_headway)
headway_mean = new_df['headway'].mean()
headway_std = new_df['headway'].std()
new_df = new_df[(new_df['headway'] >= headway_mean - 3 * headway_std) & (new_df['headway'] <= headway_mean + 3 * headway_std)]

weekly = new_df.groupby(pd.Grouper(key='Date', freq='W'))['headway_adherence'].value_counts(normalize=True)
weekly = weekly.reset_index()
weekly = weekly[weekly['headway_adherence']=='Ideal']
weekly.to_csv('weekly_counts.csv')