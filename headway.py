import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

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

def calculateHeadway():
    df_1 = pd.read_excel('historical.xlsx', sheet_name='Fall 2023') #Actually spring
    df_2 = pd.read_excel('historical.xlsx', sheet_name='Spring 2023') #Actually fall
    df_pilot = pd.read_excel('pilot_grant.xlsx', sheet_name='Sheet1') #Spring 2024
    df_3 = df_pilot.iloc[:len(df_pilot)//2]
    df_4 = df_pilot.iloc[len(df_pilot)//2:]
    results = pd.DataFrame()
    headways = []
    total_length = 0
    filtered_length = 0

    for i, og_df in enumerate([df_1, df_2, df_3, df_4]):
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
                new_df = pd.concat([new_df, df_filtered])

        new_df = new_df[~new_df['headway'].isna()]
        new_df['headway_adherence'] = new_df['headway'].apply(categorize_headway)
        total_length+=len(new_df)
        headway_mean = new_df['headway'].mean()
        headway_std = new_df['headway'].std()
        new_df = new_df[(new_df['headway'] >= headway_mean - 3 * headway_std) & (new_df['headway'] <= headway_mean + 3 * headway_std)]
        #new_df = new_df[new_df['headway'] < 100] #Remove crazy outliers
        filtered_length+=len(new_df)
        if i == 0:
            semester = "SPRING 2023"
        elif i == 1:
            semester = "FALL 2023"
        elif i == 2:
            semester = "SPRING 2024 PART 1"
        else:
            semester = "SPRING 2024 PART 2"
        print(semester)
        value_counts = new_df['headway_adherence'].value_counts(normalize=True).reindex(['Far Below Ideal', 'Below Ideal', 'Ideal', 'Above Ideal', 'Far Above Ideal'])
        print(value_counts)
        value_counts = value_counts.reset_index()
        value_counts['semester'] = semester
        results = pd.concat([results, value_counts])
        headways.append(new_df['headway'])
        # sns.violinplot(new_df['headway'])
        #
        # #new_df['headway'].hist(bins=10, edgecolor='black')
        #
        # # Add title and labels
        # plt.title('Headway Times- '+semester)
        # plt.xlabel('Headway (minutes)')
        #
        # # Show plot
        # plt.show()
    #new_df.to_csv('headway_results.csv')
    plt.boxplot(headways)
    plt.title('Headway Times Comparison')
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.xlabel('Semester')
    plt.ylabel('Headway Time')
    plt.ylim(0, 40)
    plt.xticks([1,2,3,4],['Spring 2023', 'Fall 2023', 'Spring 2024- Part 1', 'Spring 2024- Part 2'], rotation=10)
    plt.savefig('headway_comparison.png')

    for headway_list in headways:
        print(sum(headway_list) / len(headway_list))


    print(total_length)
    print(filtered_length)

    order = ['Far Below Ideal', 'Below Ideal', "Ideal", 'Above Ideal', 'Far Above Ideal']
    results['headway_adherence'] = pd.Categorical(results['headway_adherence'], categories=order, ordered=True)
    pivot_df = results.pivot(index='semester', columns='headway_adherence', values='proportion')
    pivot_df = pivot_df.reindex(['SPRING 2023', 'FALL 2023', 'SPRING 2024 PART 1', 'SPRING 2024 PART 2'])
    print(pivot_df)
    return pivot_df
pivot_df = calculateHeadway()
#pivot_df = pd.read_csv('headway_results.csv', index_col='semester')

# Plot the grouped bar chart
pivot_df.plot(kind='bar', figsize=(10, 6), color=['blue', 'lightseagreen', 'tab:green', 'gold', 'tab:red'])

# Add title and labels
plt.title('Headway Adherence')
plt.xlabel('Semester')
plt.ylabel('Proportion')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Adherence Level', bbox_to_anchor=(1,1))

# Show plot
plt.savefig('headway_adherence.png', bbox_inches='tight')