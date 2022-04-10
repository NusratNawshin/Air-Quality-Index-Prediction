#%%
import pandas as pd
df = pd.read_csv('data/pollution_2000_2021.csv')
df.head()

df = df[df['State'] == 'New York']
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)

df['Date'] = df[['Year', 'Month','Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)

df['avgAQI'] = df[['O3 AQI', 'CO AQI', 'SO2 AQI', 'NO2 AQI']].mean(axis=1)

#%%
df2 = df[df['County'] == 'Queens']
df2.reset_index(inplace=True)
df2.drop(columns=['index'], inplace=True)

#%%
df3 = df2.copy()
df3.Date=pd.to_datetime(df3.Date)
df3.set_index('Date',inplace=True)
#%%
# keeping non duplicated data from 2000-2020
data = pd.DataFrame()
for i in range(2000,2021):
    dff = df3[df3['Year'] == i]
    print(dff.shape)
    dff=dff[~dff.index.duplicated(keep='first')]
    print("--",dff.shape)
    data = pd.concat([data, dff])

# %%
# data.to_csv('data/AQI_NY_Queens.csv', index=True)
# %%

#%%
from datetime import date, timedelta

sdate = date(2000,1,1)   # start date
edate = date(2021,1,1)   # end date

dates = pd.date_range(sdate,edate-timedelta(days=1),freq='d')

datedf = pd.DataFrame(index=dates, columns=['dummy'])

# %%
merge=pd.merge(datedf,data, how='left', left_index=True, right_index=True)

# %%
