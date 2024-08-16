import pandas as pd

df = pd.read_excel('tripshot.xlsx', sheet_name='in')
df = df.sort_values(by=['day'])
print(df['headway_type'].value_counts())