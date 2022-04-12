#%%
from sympy import rotations
from toolbox import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')

from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.arima_model
from numpy import linalg as LA
import warnings
warnings.filterwarnings('ignore')

#%%
########### Description of the dataset
# a. Pre-processing dataset:

df = pd.read_csv('data/AQI_CA_LA.csv', index_col='Date')
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
plt.xticks(fontsize=16)
plt.xlabel('Time', fontsize=22)
plt.ylabel('Average Air Quality Index (AQI)', fontsize=22)
plt.tight_layout()
plt.title('Dependant Variable-avgAQI vs Time',  fontsize=30)
plt.legend(fontsize=24)
plt.grid()
plt.show()

#%% 
######### c. ACF/PACF of the dependent variable
ACF_PACF_Plot(df.avgAQI, 500)

# %%
######### d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient
# df2 = df.copy()
# df2.drop(columns=['Year','Month','Day','Address','State','City','County'], inplace=True)

sns.heatmap(df.corr(), vmin=-1,vmax=1, cmap='vlag')
plt.title('Correlation Matrix of AQI Dataset')
plt.show()
# %%
######### e. Split the dataset into train set (80%) and test set (20%)
train,test=train_test_split(df,test_size=0.2,shuffle=False)
print("Train set: ", train.shape)
print("Test set: ", test.shape)

X = df.copy()
X = X.drop(['avgAQI'], axis=1)
y = df['avgAQI']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

# %%
######### 7. Stationarity Check

# rolling mean variance
Cal_rolling_mean_var(df['avgAQI'])

print('The rolling mean is downward slopping but rolling variance is stabilizes once all samples are included.')

# %%
# ADF Test
ADF_Cal(df['avgAQI'])
print('The ADF p-value below a threshold (1% or 5%) suggests that we reject the null hypothesis and conclude that the data is stationary.')
# %%
# KPSS Test
kpss_test(df['avgAQI'])
print('The ADF p-value below a threshold (1% or 5%) suggests that we reject the null hypothesis and conclude that the data is non stationary.')

# %%
# ACF
acf(df.avgAQI,600,plot=True, title='ACF of avgAQI')
# %%
# 1st order differentiation
df2 = df.copy()

df2['avgAQI'] = df2['avgAQI'].diff(1)

Cal_rolling_mean_var(df2['avgAQI'])
ADF_Cal(df2['avgAQI'].dropna())
kpss_test(df2['avgAQI'].dropna())

print('After 1st order differenciation, the mean of dependant variable is zero\
 and both ADF and KPSS tests indicates stationarity.')
ACF_PACF_Plot(df2['avgAQI'].dropna(), 50)
# %%
# log transform then 2nd order differentiation
df3 = df.copy()
df3['avgAQI'] = df3['avgAQI'].transform(np.log).diff(1).dropna()

Cal_rolling_mean_var(df3['avgAQI'])
ADF_Cal(df3['avgAQI'].dropna())
kpss_test(df3['avgAQI'].dropna())
ACF_PACF_Plot(df3['avgAQI'].dropna(), 50)

# %%
######### 8- Time series Decomposition:

aqi = df['avgAQI']
aqi=pd.Series(np.array(df['avgAQI']),index = pd.date_range('2000-01-01',periods= len(df)))

STL = STL(aqi)
res = STL.fit()
fig = res.plot()
plt.xlabel("Time (Year)")
plt.suptitle('STL Decomposition', y=1.05)
plt.show()
# %%
T = res.trend
S = res.seasonal
R = res.resid

adj_seasonal = df['avgAQI'] - S
detrended_Temp = df['avgAQI'] - T

plt.figure()
plt.plot(T,label="Trend")
plt.plot(R,label="Residual")
plt.plot(S,label="Seasonal")
plt.xlabel("Time")
plt.ylabel("STL")
plt.legend(loc='upper right')
plt.title("Trend, Residuals and Seasonality of 'AvgAQI'")
plt.show()
# %%
F_t = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(T) + np.array(R)))

print(f"The strength of trend for this dataset is {F_t:.3f}")

# %%

F = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S) + np.array(R)))

print(f"The strength of seasonality for this datset is {F:.3f}")

# %%
print(f'Observing the graphs, trend and strength of seasonality values,\
  this dataset has low seasonality ({F:.3f}) and is trended ({F_t:.3f}).')


# %%

#seasonally adjusted data
seasonally_adj=aqi-S
#detrended data
detrended=aqi-T

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

axes[0].plot(df.index,df.avgAQI,label="Original")
axes[0].plot(df.index,seasonally_adj,label="adjusted")
axes[0].set_xlabel("Time")
axes[0].set_xticks(df.index[::1500])
axes[0].set_ylabel("Average AQI")
axes[0].set_title("Seasonality adjusted vs. Original")
axes[0].legend(loc='upper right')

axes[1].plot(df.index,df.avgAQI,label="Original")
axes[1].plot(df.index,detrended,label="Detrended")
axes[1].set_xlabel("Time")
axes[1].set_xticks(df.index[::1500])
axes[1].set_ylabel("Average AQI")
axes[1].set_title("Detrended vs. Original Data")
axes[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()
#%%
dfma = df.copy()

dfma['avgAQI_MA'] = dfma['avgAQI'].rolling(3).mean()
dfma.dropna(inplace=True)
dfma[['avgAQI', 'avgAQI_MA']].plot(label='AQI', 
                                  figsize=(16, 8))
plt.show()

# %%

moving_avg(df['avgAQI'])


# %%
######## 9. Holt-Winters method:

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

holtt = ets.ExponentialSmoothing(y_train, trend='add', damped_trend=True).fit()

holtt_predt=holtt.forecast(steps=len(y_train))
holtt_df=pd.DataFrame(holtt_predt,columns=['avgAQI']).set_index(y_train.index)

holtt_forcst=holtt.forecast(steps=len(y_test))
holtf_df=pd.DataFrame(holtt_forcst,columns=['avgAQI']).set_index(y_test.index)

plt.figure(figsize=(16,8))
plt.plot(y_train.index,y_train, label='Train', color = 'b')
plt.plot(y_test.index,y_test, label='Test', color = 'g')
plt.plot(holtt_df.index,holtt_df['avgAQI'],label='Holts winter prediction', color = 'skyblue', linestyle='dashed')
plt.plot(holtf_df.index,holtf_df['avgAQI'],label='Holts winter forecast', color = 'r')
plt.xticks(holtt_df.index.values[::1300])
plt.legend(loc = 'upper right')
plt.xlabel("Time")
plt.ylabel("Average AQI")
plt.title("Holts Winter Method")
plt.show()

# MSE
holtt_mse = mean_squared_error(y_train, holtt_df['avgAQI'])
print(f"Holt Winter Train set MSE: {holtt_mse:.3f}")

holtf_mse = mean_squared_error(y_test, holtf_df['avgAQI'])
print(f"Holt Winter Test set MSE: {holtf_mse:.3f}")

#%%
# ERRORS
res_err = y_train - holtt_df['avgAQI']
print(f"Residual Error Mean {np.mean(res_err):.3f}")
print(f"Residual Error Variance {np.var(res_err):.3f}")
fcst_err = y_test - holtf_df['avgAQI']
print(f"Forecast error Mean: {np.mean(fcst_err):.3f}")
print(f"Forecast error Variance: {np.var(fcst_err):.3f}")

acf(res_err, 50,plot=True, title='ACF of Prediction Error')
acf(fcst_err, 50,plot=True, title='ACF of Forecast Error')

# Q-Value
re = acf(res_err, 50, plot=False)
res_q = len(y_train)*np.sum(np.square(re[50+2:]))
res_q = len(y_train)*np.sum(res_q)
print(f"Q-Value of Residual Error: {res_q:.3f}")

# %%
######## 10. Feature Selection
svd_df = df.select_dtypes(include='number')
sdv_df = svd_df.drop('avgAQI', axis=1)
Y = df['avgAQI']

X_train, X_test, Y_train, Y_test = train_test_split(svd_df,Y, shuffle=False, test_size=0.2)

Xx = sm.add_constant(X_train.values)
H = np.matmul(Xx.T, Xx)
_,d,_ = np.linalg.svd(H)
print(f'singular value for X are :- \n{pd.DataFrame(d)}')
# Any singular values close to zero means that one or more
# features are correlated. The correlated feature(s) needs to be
# detected and removed from the feature space.

print('\nAt least 4 features are correlated as their singular values is zero.')

print(f'\nThe condition number for x is {LA.cond(Xx):.2f}') # its k -> if small <100 then good
# The κ < 100 =⇒ Weak Degree of Co-linearity(DOC)
# • The 100 < κ < 1000 =⇒ Moderate to Strong DOC
# • The κ > 1000 =⇒ Severe DOC

print("\nAs the condition number is very high there is a severe Degree of Colinearity.")

#%%
# OLS

# %%
######## 11. Base model
# average
# naive
# drift
# SES
# 

# %%
######## 12. Multiple Linear Regression

# %%
######## 13. ARMA, ARIMA SARIMA

# %%
######## 14. LMA

# %%
######## 15. Diagnostic Analysis


# %%
######## 16. Deep learning Model LSTM 

# %%
######## 17. Final model Selection

# %%
######## 18. Forecast Function

# %%
######## 19. h-step Prediction

# %%
######## 20. Summary  and Conclusion







# %%
