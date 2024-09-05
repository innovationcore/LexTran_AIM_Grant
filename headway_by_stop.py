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

df_1 = pd.read_excel('historical.xlsx', sheet_name='Fall 2023') #Actually spring
df_2 = pd.read_excel('historical.xlsx', sheet_name='Spring 2023') #Actually fall
df_3 = pd.read_excel('pilot_grant.xlsx', sheet_name='Sheet1') #Spring 2024
new_df = pd.DataFrame()
for og_df in [df_1, df_2, df_3]:
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
new_df = new_df[['Stop_Name', 'headway_adherence']]
custom_category_order = ['Far Below Ideal', 'Below Ideal', 'Ideal', 'Above Ideal', 'Far Above Ideal']
new_df['headway_adherence'] = pd.Categorical(new_df['headway_adherence'], categories=custom_category_order, ordered=True)
values = new_df.groupby('Stop_Name')['headway_adherence'].value_counts(normalize=True)
values_df = values.reset_index(name='proportion')
ideal_values = values_df[values_df['headway_adherence'] == 'Ideal']

sorted_ideal_values = ideal_values.sort_values(by='proportion', ascending=True)
sorted_stop_names = sorted_ideal_values['Stop_Name']
sorted_values = values.reindex(sorted_stop_names, level='Stop_Name').reset_index()
sorted_values['Stop_Name'] = pd.Categorical(sorted_values['Stop_Name'], categories=sorted_stop_names, ordered=True)
#sorted_values = sorted_values.sort_index(level='headway_adherence', key=lambda x: pd.Categorical(x, categories=custom_category_order, ordered=True))
sorted_values['headway_adherence'] = pd.Categorical(sorted_values['headway_adherence'], categories=custom_category_order, ordered=True)
sorted_values = sorted_values.sort_values(by=['Stop_Name','headway_adherence'])
print(sorted_values)
sorted_values.to_csv('headway_by_stop.csv', index=False)
