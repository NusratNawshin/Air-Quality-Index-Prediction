#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('data/AQI_NY_Queens.csv', index_col='Date')
df.head()
# %%
plt.figure(figsize=(12,8))
# plt.plot(df['avgAQI'][:50])
df['avgAQI'].plot()
plt.xticks(rotation =90)
plt.show()
# %%
