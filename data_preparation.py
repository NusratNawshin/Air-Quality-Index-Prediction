#%%
import pandas as pd
df = pd.read_csv('data/pollution_2000_2021.csv')
df.head()

#%% 
# keeping only California data 
df = df[df['State'] == 'California']
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)

#%%

print(df.shape)

#%%
# making date column by merging Year, month and day
df['Date'] = df[['Year', 'Month','Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)

#%% 
# creating target coulmn avgAQI from 4 AQI data columns
df['avgAQI'] = df[['O3 AQI', 'CO AQI', 'SO2 AQI', 'NO2 AQI']].mean(axis=1)

#%%
# keeping only Los Angeles county for further analysis
df2 = df[df['County'] == 'Los Angeles']
df2.reset_index(inplace=True)
df2.drop(columns=['index'], inplace=True)

#%%
# making date as index
df3 = df2.copy()
df3.Date=pd.to_datetime(df3.Date)
df3.set_index('Date',inplace=True)

# keeping non duplicated data 
finaldf=df3[~df3.index.duplicated(keep='first')]
print(df.shape)
print(finaldf.Year.value_counts().sort_index())


#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
finaldf['avgAQI'].plot()
plt.xlabel('Time', fontsize=22)
plt.ylabel('Average Air Quality Index (AQI)', fontsize=22)
plt.tight_layout()
plt.title('Dependant Variable-avgAQI vs Time',  fontsize=30)
plt.legend(fontsize=24)
plt.show()


#%%
# storing dataframe in csv
# finaldf.to_csv('data/AQI_CA_LA.csv', index=True, index_label='Date')
# %%
