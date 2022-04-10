#%%
from cProfile import label
from toolbox import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

#%%
########### a. Pre-processing dataset:

df = pd.read_csv('data/AQI_NY_Queens.csv', index_col='Date')
print('This dataset is about the Air Quality Index of New York Queens county.')
print("1st 5 values of the dataset: \n",df.head())
#%%
print('Details of dataset: \n')
print(df.info())

print(f"The dataset contains {df.shape[0]} number of rows and \
{df.shape[1]} columns and doesn't contain any missing values.\
It has the AQI data from {df.index[0]} to {df.index[-1]}. ")


# %%
########## b. plotting dependent variable vs time.

print("For this time series analysis, my dependant variable is 'avgAQI'\
 which is the the average Air Quality Index of O3, CO, SO2 and NO2. ")

plt.figure(figsize=(14,8))
# plt.plot(df.index, df['avgAQI'], label='Dependant Variable-avgAQI')
df['avgAQI'].plot()
# plt.xticks(ticks=df.index, labels = df.Year, fontsize=16)
plt.xlabel('Time', fontsize=22)
plt.ylabel('Average Air Quality Index (AQI)', fontsize=22)
plt.tight_layout()
plt.title('Dependant Variable-avgAQI vs Time',  fontsize=30)
plt.legend(fontsize=24)
# plt.xticks(rotation =90)
plt.show()

#%% 
######### c. ACF/PACF of the dependent variable

# %%
df2 = df.copy()
df2.drop(columns=['Year','Month','Day','Address','State','City','County'], inplace=True)

sns.heatmap(df2.corr())
plt.show()
# %%

# %%
