import pandas as pd

df = pd.read_csv('Lextran_InstanceData_20240207_20240503.csv')
df = df[df['Day'] == '2024/02/07']
df = df[df['Shift']=='14-1']

df2 = pd.read_excel('pilot_grant.xlsx', sheet_name='Sheet1')
df2 = df2[df2['Date'] == '02/07/2024']
df2 = df2[df2['Route']==141]
df2 = df2.sort_values(by=['Arrival'])
df2 = df2[df2['Arrival']>'2024-02-07 06:00:00']

stop_names = {'West Blue Lot':'West Blue Lot',
              'Garrigus Building':'Garrigus Bldg',
              'Ag North':'Ag North',
              'Chandler Hospital':'ChandlerHospita',
              'University Health':'UniversityHealt',
              'College of Law':'College of Law',
              'Memorial Hall':'Memorial Hall',
              'Main Building':'Main Bldg',
              'Student Center':'Student Center',
              'Singletary Center':'Singletary Ctr',
              'College of Fine Arts':'CollegeFineArt',
              'Columbia Avenue':'Columbia Ave',
              'Woodland Avenue':'Woodland Ave',
              'W. T. Young Library':'WTYoungLibrary',
              'Huguelet Drive':'Huguelet Dr',
              'Hospital Drive':'Hospital Dr',
              'Tobacco Research':'TobaccoResearch'}
df['Stop_Name'] = df['Stop'].map(stop_names)

df['stop_count'] = df.groupby('Stop_Name').cumcount()+1
df2['stop_count'] = df2.groupby('Stop_Name').cumcount()+1
print(df['stop_count'].max())
print(df2['stop_count'].max())
merged = pd.merge(df, df2, on=['Stop_Name','stop_count'])
merged = merged.drop(columns=['stop_count'])
merged = merged[['Date','Trip','Stop_Name','Arrival','Dwell (s)','Departure','Scheduled Time']]
merged.to_csv('merged.csv')