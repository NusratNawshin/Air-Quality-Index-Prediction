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
from numpy import linalg as LA
import warnings
warnings.filterwarnings('ignore')

#%%
########### Description of the dataset
# a. Pre-processing dataset:

df = pd.read_csv('data/AQI_CA_LA.csv', index_col='Date')
print('This dataset is about the Air Quality Index of California Loss Angeles county.')
print("1st 5 values of the dataset: \n",df.head())

#%%
print('Details of dataset: \n')
print(df.info())

print(f"The dataset contains {df.shape[0]} number of rows and \
{df.shape[1]} columns and doesn't contain any missing values.\
It has the AQI data from {df.index[0]} to {df.index[-1]}. ")

#%%
print(df.describe())

# %%
########## b. plotting dependent variable vs time.

print("For this time series analysis, my dependant variable is 'avgAQI'\
 which is the the average Air Quality Index of O3, CO, SO2 and NO2. ")

plt.figure(figsize=(14,8))
plt.plot(df.index, df['avgAQI'], label='Dependant Variable-avgAQI')
plt.xticks(df.index[::981])
plt.xlabel('Time', fontsize=22)
plt.ylabel('Average Air Quality Index (AQI)', fontsize=22)
plt.tight_layout()
plt.title('Dependant Variable-avgAQI vs Time',  fontsize=30)
plt.legend(fontsize=24)
plt.grid()
plt.show()

#%% 
######### c. ACF/PACF of the dependent variable
ACF_PACF_Plot(df.avgAQI, 100)

# %%
######### d. Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient
# df2 = df.copy()
# df2.drop(columns=['Year','Month','Day','Address','State','City','County'], inplace=True)
dff = df.drop(['O3 AQI','CO AQI','SO2 AQI','NO2 AQI'], axis=1)

plt.figure(figsize=(14,14))

sns.heatmap(dff.corr(), vmin=-1,vmax=1,cmap='RdBu_r', annot=True)

plt.title('Correlation Matrix of AQI Dataset')
plt.show()
# %%
######### e. Split the dataset into train set (80%) and test set (20%)
# train,test=train_test_split(df,test_size=0.2,shuffle=False)
# print("Train set: ", train.shape)
# print("Test set: ", test.shape)

X = df.copy()
X = X.drop(['avgAQI', 'O3 AQI','CO AQI','SO2 AQI','NO2 AQI'], axis=1)
y = df['avgAQI']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
print("Train set: ", x_train.shape)
print("Test set: ", x_test.shape)


# %%
######### 7. Stationarity Check

# original dataset-rolling mean variance
print('Original dataset-rolling mean & variance')
Cal_rolling_mean_var(df['avgAQI'])

print('The rolling mean is downward slopping but rolling variance is stabilizes once all samples are included.')

# %%
# ADF Test
ADF_Cal(df['avgAQI'])
print('The ADF p-value below a threshold (1% or 5%) suggests that we reject the null hypothesis and conclude that the data is stationary.')
# %%
# KPSS Test
kpss_test(df['avgAQI'])
print('The KPSS p-value below a threshold (1% or 5%) suggests that we reject the null hypothesis and conclude that the data is non stationary.')


# %%
# ACF
acf(df.avgAQI,100,plot=True, title='ACF of avgAQI')
# %%
# 1st order differentiation
# df2 = df.copy()

# df2['avgAQI'] = df2['avgAQI'].diff(1)

# Cal_rolling_mean_var(df2['avgAQI'])
# ADF_Cal(df2['avgAQI'].dropna())
# kpss_test(df2['avgAQI'].dropna())

# print('After 1st order differenciation, the mean of dependant variable is zero\
#  and both ADF and KPSS tests indicates stationarity.')
# ACF_PACF_Plot(df2['avgAQI'].dropna(), 100)
# %%
# log transform then 1nd order differentiation
# df3 = df.copy()
# df3['avgAQI'] = df3['avgAQI'].transform(np.log).diff(1).dropna()

# Cal_rolling_mean_var(df3['avgAQI'])
# ADF_Cal(df3['avgAQI'].dropna())
# kpss_test(df3['avgAQI'].dropna())
# ACF_PACF_Plot(df3['avgAQI'].dropna(), 50)

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
# Strength of trend
F_t = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(T) + np.array(R)))

print(f"The strength of trend for this dataset is {F_t:.3f}")

# %%
# Strength of seasonality
F = np.maximum(0, 1 - np.var(np.array(R)) / np.var(np.array(S) + np.array(R)))

print(f"The strength of seasonality for this dataset is {F:.3f}")

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
# Moving Average (3)
dfma = df.copy()

dfma['avgAQI_MA'] = dfma['avgAQI'].rolling(3).mean()
dfma.dropna(inplace=True)
dfma[['avgAQI', 'avgAQI_MA']].plot(label='AQI', 
                                  figsize=(16, 8))
plt.title('3-MA')
plt.grid()
plt.show()


# %%
######## 9. Holt-Winters method:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

holtt = ets.ExponentialSmoothing(y_train,  trend='add', damped_trend=True, seasonal=None).fit()

# holtt_predt=holtt.forecast(steps=len(y_train))
holtt_predt=holtt.fittedvalues
holtt_df=pd.DataFrame(holtt_predt,columns=['avgAQI']).set_index(y_train.index)

holtt_forcst=holtt.forecast(steps=len(y_test))
holtf_df=pd.DataFrame(holtt_forcst,columns=['avgAQI']).set_index(y_test.index)

plt.figure(figsize=(16,8))
plt.plot(y_train.index,y_train, label='Train', color = 'b')
plt.plot(y_test.index,y_test, label='Test', color = 'g')
# plt.plot(holtt_df.index,holtt_df['avgAQI'],label='Holts winter prediction', color = 'skyblue', linestyle='dashed')
plt.plot(holtf_df.index,holtf_df['avgAQI'],label='Holts winter forecast', color = 'r')
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.legend(loc = 'upper right')
plt.xlabel("Time")
plt.ylabel("Average AQI")
plt.title("Holts Winter Method")
plt.grid()
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
# re = acf(res_err, 50, plot=False)
# res_q = len(y_train)*np.sum(np.square(re[50+2:]))
# res_q = len(y_train)*np.sum(res_q)
res_q = q_value(res_err, 50, len(y_train))
print(f"Q-Value of Residual Error: {res_q:.3f}")

#%%
qstar,pvalue=sm.stats.acorr_ljungbox(res_err,lags=[50])


# print(f"Q*-Value of Residual Error: {qstar[0]:.3f}")

# if res_q < qstar:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


# %%
######## 10. Feature Selection

svd_df = df.select_dtypes(include='number')
svd_df.drop(['O3 AQI','CO AQI','SO2 AQI','NO2 AQI'], axis=1, inplace=True)
sdv_df = svd_df.drop('avgAQI', axis=1)
X = sm.add_constant(sdv_df)
Y = df['avgAQI']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=False, test_size=0.2)

Xx = X.values
H = np.matmul(Xx.T, Xx)
_,d,_ = np.linalg.svd(H)
print(f'singular value for X are :- \n{pd.DataFrame(d)}')
# Any singular values close to zero means that one or more
# features are correlated. The correlated feature(s) needs to be
# detected and removed from the feature space.

print('\nAt least 4 features are correlated as their singular values is closer to zero.')

print(f'\nThe condition number for x is {LA.cond(Xx):.2f}') # its k -> if small <100 then good
# The κ < 100 =⇒ Weak Degree of Co-linearity(DOC)
# • The 100 < κ < 1000 =⇒ Moderate to Strong DOC
# • The κ > 1000 =⇒ Severe DOC

print("\nAs the condition number is very high there is a severe Degree of Co-linearity.")

#%%

print("I am using PCA for feature elimination.")

from sklearn.decomposition import PCA
pca=PCA(n_components='mle',svd_solver='full')
pca.fit(sdv_df)
aqi_pca=pca.transform(sdv_df)

print("Original Dimension:",sdv_df.shape)
print("Transformed dimension:", aqi_pca.shape)
print("Explained variance ratio:\n",pca.explained_variance_ratio_)
x=np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs Number of Components")
plt.grid()
plt.show()

print('PCA already reduced the features from 15 to 14, but\
 more feature can be removed as with 5 just features we are getting\
 more than 90% explained variance. ')

# Making new reduced feature space with 5 components
pcaf=PCA(n_components=5,svd_solver='full')
pcaf.fit(sdv_df)
reduced_aqi_pcaf=pcaf.transform(sdv_df)

print("\nOriginal Dimension:",sdv_df.shape)
print("Transformed Dimension:",reduced_aqi_pcaf.shape)
print("Explained variance ratio:\n",pcaf.explained_variance_ratio_)


x=np.arange(1,len(np.cumsum(pcaf.explained_variance_ratio_))+1,1)
plt.plot(x,np.cumsum(pcaf.explained_variance_ratio_))
plt.xticks(x)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance Vs Number of Components")
plt.suptitle("Reduced Feature Space", fontsize = 22)
plt.grid()
plt.show()


PlayStore_pcaf_df=pd.DataFrame(reduced_aqi_pcaf).corr()
column=[]
for i in range(reduced_aqi_pcaf.shape[1]):
    column.append(f'Pricipal Component {i+1}')
plt.figure(figsize=(8,6))
sns.heatmap(PlayStore_pcaf_df,annot=True, xticklabels=column,yticklabels=column)
plt.title("Correlation Coefficient of Reduced Feature Space")
plt.show()


#%%
# OLS
# Full model
model=sm.OLS(Y_train,X_train).fit()
print(model.summary())
#%%
# maximum p of model 1 
max_p=pd.DataFrame(model.pvalues,columns=['P_Values']).set_index(model.pvalues.index)
max_p = max_p[max_p['P_Values'] == max_p['P_Values'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P-value is on column 'Month'.Let's drop this feature\
 from the train set and rebuilt the model.")

#%%
# model 2
X_train.drop(['Month'], axis=1, inplace=True)
model2 = sm.OLS(Y_train,X_train).fit()
print('Model 2: After dropping Month: \n',model2.summary())

#%%
# max p
max_p=pd.DataFrame(model2.pvalues,columns=['P_Values']).set_index(model2.pvalues.index)
max_p = max_p[max_p['P_Values'] == max_p['P_Values'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P-value is on column 'SO2 1st Max Hour'.Let's drop this feature\
 from the train set and build the 3rd model.")

#%%
# model 3
X_train.drop(['SO2 1st Max Hour'], axis=1, inplace=True)
model3 = sm.OLS(Y_train,X_train).fit()
print('Model 3: After dropping SO2 1st Max Hour: \n',model3.summary())

#%%
# max p_values
max_p=pd.DataFrame(model3.pvalues,columns=['p_values']).set_index(model3.pvalues.index)
max_p = max_p[max_p['p_values'] == max_p['p_values'].max()]
print(f"\nMaximum p_values: \n{max_p}")

print("The highest p_values is on column 'Day'.Let's drop this feature\
 from the train set and build the 4th model.")

#%%
# Model 4
X_train.drop(['Day'], axis=1, inplace=True)
model4 = sm.OLS(Y_train,X_train).fit()
print('Model 4: After dropping Day: \n',model4.summary())

#%%
# max p
max_p=pd.DataFrame(model4.pvalues,columns=['P_Value']).set_index(model4.pvalues.index)
max_p = max_p[max_p['P_Value'] == max_p['P_Value'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P-value is on column 'O3 1st Max Hour'.Let's drop this feature\
 from the train set and build the 5th model.")

#%%
# Model 5
X_train.drop(['O3 1st Max Hour'], axis=1, inplace=True)
model5 = sm.OLS(Y_train,X_train).fit()
print('Model 5: After dropping O3 1st Max Hour: \n',model5.summary())

#%%
# max p
max_p=pd.DataFrame(model5.pvalues,columns=['P_Value']).set_index(model5.pvalues.index)
max_p = max_p[max_p['P_Value'] == max_p['P_Value'].max()]
print(f"\nMaximum P_Value: \n{max_p}")

print("The highest P_Value is on column 'NO2 1st Max Hour'.Let's drop this feature\
 from the train set and build the 6th model.")

#%%
# Model 6
X_train.drop(['NO2 1st Max Hour'], axis=1, inplace=True)
model6 = sm.OLS(Y_train,X_train).fit()
print('Model 6: After dropping NO2 1st Max Hour: \n',model6.summary())

#%%
# max p_values
max_p=pd.DataFrame(model6.pvalues,columns=['p_values']).set_index(model6.pvalues.index)
max_p = max_p[max_p['p_values'] == max_p['p_values'].max()]
print(f"\nMaximum p_values: \n{max_p}")

print("The highest p_values is on column 'CO 1st Max Hour'.Let's drop this feature\
 from the train set and build the 7th model.")

#%%
# Model 7

X_train.drop(['CO 1st Max Hour'], axis=1, inplace=True)
model7 = sm.OLS(Y_train,X_train).fit()
print('Model 7: After dropping CO 1st Max Hour: \n',model7.summary())

#%%
# max P_Value
max_p=pd.DataFrame(model7.pvalues[1:],columns=['P_Value']).set_index(model7.pvalues.index[1:])
max_p = max_p[max_p['P_Value'] == max_p['P_Value'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P_Value is on column 'Year'.Let's drop this feature\
 from the train set and buil the 8th model.")

#%%
# Model 8
X_train.drop(['Year'], axis=1, inplace=True)
model8 = sm.OLS(Y_train,X_train).fit()
print('Model 8: After dropping Year: \n',model8.summary())

#%%
# max p_value
max_p=pd.DataFrame(model8.pvalues,columns=['p_value']).set_index(model8.pvalues.index)
max_p = max_p[max_p['p_value'] == max_p['p_value'].max()]
print(f"\nMaximum p_value: \n{max_p}")

print("The highest p_value is on column 'CO Mean'.Let's drop this feature\
 from the train set and build the 9th model.") 

#%%
# Model 9
X_train.drop(['CO Mean'], axis=1, inplace=True)
model9 = sm.OLS(Y_train,X_train).fit()
print('Model 9: After dropping CO Mean: \n',model9.summary())

#%%
# max p
max_p=pd.DataFrame(model9.pvalues,columns=['P_Value']).set_index(model9.pvalues.index)
max_p = max_p[max_p['P_Value'] == max_p['P_Value'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P-value is on column 'SO2 Mean'.Let's drop this feature\
 from the train set and build the 10th model.") 

#%%
# Model 10
X_train.drop(['SO2 Mean'], axis=1, inplace=True)
model10 = sm.OLS(Y_train,X_train).fit()
print('Model 10: After dropping SO2 Mean: \n',model10.summary())

#%%
# max p
max_p=pd.DataFrame(model10.pvalues,columns=['P_Value']).set_index(model10.pvalues.index)
max_p = max_p[max_p['P_Value'] == max_p['P_Value'].max()]
print(f"\nMaximum p-value: \n{max_p}")

print("The highest P-value is on column 'O3 Mean'.Let's drop this feature\
 from the train set and build the 11th model.") 

#%%
# Model 11
X_train.drop(['O3 Mean'], axis=1, inplace=True)
model11 = sm.OLS(Y_train,X_train).fit()
print('Model 11: After dropping O3 Mean: \n',model11.summary())

#%%
# max p_value
max_p=pd.DataFrame(model11.pvalues,columns=['p_value']).set_index(model11.pvalues.index)
max_p = max_p[max_p['p_value'] == max_p['p_value'].max()]
print(f"\nMaximum p_value: \n{max_p}")

print("The highest p_value is on column 'NO2 Mean'.Let's drop this feature\
 from the train set and build the 12th model.")  

#%%
# Model 12
X_train.drop(['NO2 Mean'], axis=1, inplace=True)
model12 = sm.OLS(Y_train,X_train).fit()
print('Model 12: After dropping NO2 Mean: \n',model12.summary())

#%%
# max p_value
max_p=pd.DataFrame(model12.pvalues,columns=['p_value']).set_index(model12.pvalues.index)
max_p = max_p[max_p['p_value'] == max_p['p_value'].max()]
print(f"\nMaximum p_value: \n{max_p}")

print("The highest p_value is on column 'SO2 1st Max Value'.Let's drop this feature\
 from the train set and build the 13th model.")  

#%%
# Model 13
X_train.drop(['SO2 1st Max Value'], axis=1, inplace=True)
model13 = sm.OLS(Y_train,X_train).fit()
print('Model 13: After dropping SO2 1st Max Value: \n',model13.summary())

#%%
# max p_value
max_p=pd.DataFrame(model13.pvalues,columns=['p_value']).set_index(model13.pvalues.index)
max_p = max_p[max_p['p_value'] == max_p['p_value'].max()]
print(f"\nMaximum p_value: \n{max_p}")

print('All the pvalues are 0 but the condition number is high. Lets drop the highest std err O3 1st Max Value')

#%%
# Model 14
X_train2 = X_train.copy()
X_train2.drop(['O3 1st Max Value'], axis=1, inplace=True)
model14 = sm.OLS(Y_train,X_train2).fit()
print('Model 14: After dropping O3 1st Max Value: \n',model14.summary())

#%%

print('After dropping O3 1st Max Value, the adjusted r squared drops a lot. So instead of dropping O3 1st Max Value, lets drop CO 1st Max Value')

#%%
# Model 15
X_train3 = X_train.copy()
X_train3.drop(['CO 1st Max Value'], axis=1, inplace=True)
model15 = sm.OLS(Y_train,X_train3).fit()
print('Model 15: After dropping CO 1st Max Value: \n',model15.summary())


print('Now by dropping CO 1st Max Value, the condition number is still high. So instead of dropping CO 1st Max Value, lets drop NO2 1st Max Value')

#%%
# Model 16
X_train4 = X_train.copy()
X_train4.drop(['NO2 1st Max Value'], axis=1, inplace=True)
model16 = sm.OLS(Y_train,X_train4).fit()
print('Model 16: After dropping NO2 1st Max Value: \n',model16.summary())

#%% 
# Final model - model16
print(f"Now the high condition number is reduced and the final mmodel contains only 2 features,\
 O3 1st Max Value and CO 1st Max Value. The adjusted r squared is  {((model16.rsquared_adj)*100):2f}%")

X_train.drop(['NO2 1st Max Value'], axis=1, inplace=True)
X_test = X_test[['const','O3 1st Max Value','CO 1st Max Value']]
#%%
final_model = sm.OLS(Y_train,X_train).fit()

# %%
######## 11. Base model
# average
# 1 - step
# df_avg = pd.DataFrame(data = {'yt':Y_train})
df_avg = pd.DataFrame(data = {'yt':y_train})
df_avg['y_hat'] = avg_pred(df_avg.index,df_avg['yt'])
df_avg['e'] = df_avg['yt']-df_avg['y_hat']
df_avg['e^2'] = round(df_avg['e']**2, 2)

# h - step
# df_avg_h = pd.DataFrame(data = {'yt+h':Y_test})
df_avg_h = pd.DataFrame(data = {'yt+h':y_test})
df_avg_h['y_hat'] = np.mean(df_avg['yt'])
df_avg_h['e'] = df_avg_h['yt+h']-df_avg_h['y_hat']
df_avg_h['e^2'] = round(df_avg_h['e']**2, 2)

plt.figure(figsize=(16,6))
plt.plot(df_avg.index,df_avg['yt'], label='Training Dataset', color='b')
plt.plot(df_avg_h.index,df_avg_h['yt+h'], label = 'Testing Dataset', color='g')
plt.plot(df_avg_h.index,df_avg_h['y_hat'], label = 'Avg Method H-step prediction', color='r', linestyle='dashed')
# plt.xticks(df.index[::981])
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.title('Average Method & Forecast')
plt.xlabel('time')
plt.ylabel('Values')
plt.legend()
plt.show()

# MSE of prediction
MSE_pred = round(np.mean(df_avg['e^2']),2)
print("MSE of prediction: ", MSE_pred)

# MSE of forecast
MSE_forecast = round(np.mean(df_avg_h['e^2']),2)
print("MSE of forecast: ", MSE_forecast)

# Variance error of prediction
var_err_pred = round(np.var(df_avg['e']),2)
print("Variance of prediction error: ", var_err_pred)

# Variance error of Forecast
var_err_forecast = round(np.var(df_avg_h['e']),2)
print("Variance of Forecast error: ", var_err_forecast)

# ACF
acf(df_avg['e'], 50, plot= True, title="ACF of Average Method Residuals")

# Prediction Q
q_avg = q_value(df_avg['e'],50,len(df_avg))
print(f"Q-Value: {q_avg:.2f}")

qstar_avg,pvalue_avg=sm.stats.acorr_ljungbox(df_avg['e'],lags=[50])

# print(f"Q*-Value of Residual Error: {qstar_avg[0]:.2f}")

# if res_q < qstar_avg:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue_avg > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


#%%
# Naive

# df_niv = pd.DataFrame(data = {'yt':Y_train})
df_niv = pd.DataFrame(data = {'yt':y_train})
df_niv['y_hat'] = naive_forecast(np.arange(0,len(df_niv.index)),df_niv['yt'])
df_niv['e'] = df_niv['yt']-df_niv['y_hat']
df_niv['e^2'] = round(df_niv['e']**2, 2)

# h - step
# df_niv_h = pd.DataFrame(data = {'yt+h':Y_test})
df_niv_h = pd.DataFrame(data = {'yt+h':y_test})
df_niv_h['y_hat'] = df_niv['yt'].iloc[-1]
df_niv_h['e'] = df_niv_h['yt+h']-df_niv_h['y_hat']
df_niv_h['e^2'] = round(df_niv_h['e']**2, 2)

plt.figure(figsize=(16,6))
plt.plot(df_niv.index,df_niv['yt'], label='Training Dataset', color='b')
plt.plot(df_niv_h.index,df_niv_h['yt+h'], label = 'Testing Dataset', color='g')
plt.plot(df_niv_h.index,df_niv_h['y_hat'], label = 'Naive Method H-step prediction', color='r', linestyle='dashed')
# plt.xticks(df.index[::981])
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.title('Naive Method & Forecast')
plt.xlabel('time')
plt.ylabel('Values')
plt.legend()
plt.show()

# MSE of prediction
MSE_pred_nv = round(np.mean(df_niv['e^2']),2)
print("MSE of prediction: ", MSE_pred_nv)

# MSE of forecast
MSE_forecast_nv = round(np.mean(df_niv_h['e^2']),2)
print("MSE of forecast: ", MSE_forecast_nv)

# Variance error of prediction
var_err_pred_nv = round(np.var(df_niv['e']),2)
print("Variance of prediction error: ", var_err_pred_nv)

# Variance error of Forecast
var_err_forecast_nv = round(np.var(df_niv_h['e']),2)
print("Variance of Forecast error: ", var_err_forecast_nv)

# ACF
acf(df_niv['e'], 50, plot= True, title="ACF of Naive Method Residuals")

# Prediction Q
q_niv = q_value(df_niv['e'],50,len(df_niv))
print(f"Q-Value: {q_niv:.2f}")

qstar_niv,pvalue_niv=sm.stats.acorr_ljungbox(df_niv['e'][1:],lags=[50])

# print(f"Q*-Value of Residual Error: {qstar_niv[0]:.2f}")

# if q_niv < qstar_niv:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue_niv > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


#%%
# Drift

# df_drft = pd.DataFrame(data = {'yt':Y_train})
df_drft = pd.DataFrame(data = {'yt':y_train})
df_drft['y_hat'] = drift_predict(np.arange(0,len(df_drft.index)),df_drft['yt'],1)
df_drft['e'] = df_drft['yt']-df_drft['y_hat']
df_drft['e^2'] = round(df_drft['e']**2, 2)

# h - step
# df_drft_h = pd.DataFrame(data = {'yt+h':Y_test})
df_drft_h = pd.DataFrame(data = {'yt+h':y_test})
df_drft_h['y_hat'] = drift_forecast(Y_train[0],Y_train[-1],len(Y_train),len(df_drft_h))
df_drft_h['e'] = df_drft_h['yt+h']-df_drft_h['y_hat']
df_drft_h['e^2'] = round(df_drft_h['e']**2, 2)


plt.figure(figsize=(16,6))
plt.plot(df_drft.index,df_drft['yt'], label='Training Dataset', color='b')
plt.plot(df_drft_h.index,df_drft_h['yt+h'], label = 'Testing Dataset', color='g')
plt.plot(df_drft_h.index,df_drft_h['y_hat'], label = 'Drift Method H-step prediction', color='r', linestyle='dashed')
# plt.xticks(df.index[::981])
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.title('Drift Method & Forecast')
plt.xlabel('time')
plt.ylabel('Values')
plt.legend()
plt.show()

# MSE of prediction
MSE_pred_df = round(np.mean(df_drft['e^2']),2)
print("MSE of prediction: ", MSE_pred_df)

# MSE of forecast
MSE_forecast_df = round(np.mean(df_drft_h['e^2']),2)
print("MSE of forecast: ", MSE_forecast_df)

# Variance error of prediction
var_err_pred_df = round(np.var(df_drft['e']),2)
print("Variance of prediction error: ", var_err_pred_df)

# Variance error of Forecast
var_err_forecast_df = round(np.var(df_drft_h['e']),2)
print("Variance of Forecast error: ", var_err_forecast_df)

# ACF
acf(df_drft['e'], 50, plot= True, title="ACF of Drift Method Residuals")


# Prediction Q
q_dft = q_value(df_drft['e'],50,len(df_drft))
print(f"Q-Value: {q_dft:.2f}")

qstar_dft,pvalue_dft=sm.stats.acorr_ljungbox(df_drft['e'][2:],lags=[50])

# print(f"Q*-Value of Residual Error: {qstar_dft[0]:.2f}")

# if q_dft < qstar_dft:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue_dft > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


#%%
# Simple Exponential Smoothing (SES)

ses = ets.ExponentialSmoothing(y_train, trend =None, damped_trend = False, seasonal = None).fit()
ses_pred = pd.DataFrame(ses.fittedvalues).set_index(y_train.index)
ses_frcst = pd.DataFrame(ses.forecast(steps=len(y_test))).set_index(y_test.index)

ses_pred_err = y_train - ses_pred[0].values
ses_frcst_err = y_test - ses_frcst[0].values

plt.figure(figsize=(16,6))
# plt.plot(Y_train.index,Y_train.values, label='Training Dataset', color='b')
plt.plot(y_train.index,y_train.values, label='Training Dataset', color='b')
# plt.plot(Y_test.index,Y_test, label = 'Testing Dataset', color='g')
plt.plot(y_test.index,y_test, label = 'Testing Dataset', color='g')
plt.plot(ses_frcst.index,ses_frcst[0].values, label = 'SES Method H-step prediction', color='r', linestyle='dashed')
# plt.xticks(df.index[::981])
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.title('Simple Exponential Smoothing Method & Forecast')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend(loc='upper right')
plt.show()

# MSE of prediction
# MSE_ses_pred = mean_squared_error(Y_train,ses_pred)
MSE_ses_pred = mean_squared_error(y_train,ses_pred)
print(f"MSE of prediction: {MSE_ses_pred:.3f}")

# MSE of forecast
# MSE_ses_frcst = mean_squared_error(Y_test, ses_frcst)
MSE_ses_frcst = mean_squared_error(y_test, ses_frcst)
print(f"MSE of forecast: {MSE_ses_frcst:.3f}")

# Variance error of prediction
var_err_ses_pred = round(np.var(ses_pred_err),2)
print(f"Variance of prediction error: {var_err_ses_pred:.3f}")

# Variance error of Forecast
var_err_ses_frcst = round(np.nanvar(ses_frcst_err),2)
print(f"Variance of Forecast error: {var_err_ses_frcst:.3f}")

# ACF
acf(ses_pred_err, 50, plot= True, title="ACF of SES Method Residuals")


# Prediction Q
q_ses = q_value(ses_pred_err,50,len(ses_pred_err))
print(f"Q-Value: {q_ses:.2f}")

qstar_ses,pvalue_ses=sm.stats.acorr_ljungbox(ses_pred_err,lags=[50])

# print(f"Q*-Value of Residual Error: {qstar_ses[0]:.2f}")

# if q_ses < qstar_ses:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue_ses > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


# %%
######## 12. Multiple Linear Regression

print('From the backward stepwise feature selection we got our Final Multiple Linear Regression model.')
# final_model = sm.OLS(Y_train,X_train).fit()
print('Final Model: \n', final_model.summary())
#%%
# Predictions
# pred=final_model.predict(X_train)
pred = final_model.fittedvalues

#Residual error
res_err=Y_train-pred

#Forecasts
forecast=final_model.predict(X_test)

#Forecast error
fcst_err=Y_test-forecast
#%%
# Prediction Plot
plt.figure(figsize=(16,6))
Y_train.plot(label='Train set')
plt.plot(pred.index,pred, label='Predicted')
plt.title('AvgAQI Prediction using OLS model')
plt.ylabel('AQI')
plt.xlabel('Time')
plt.grid()
plt.legend(loc='upper right')
plt.show()
# Forecast Plot
plt.figure(figsize=(16,6))
Y_test.plot(label='Test set')
plt.plot(forecast.index,forecast, label='Forecast')
plt.title('AvgAQI Forecast using OLS model')
plt.ylabel('AQI')
plt.xlabel('Time')
plt.grid()
plt.legend(loc='upper right')
plt.show()
# together
plt.figure(figsize=(16,6))
Y_train.plot(label='Train set', color = 'b')
plt.plot(pred.index,pred, label='Predicted', color = 'deepskyblue')
plt.plot(Y_test, label='Test', color = 'maroon')
plt.plot(forecast.index,forecast, label='Forecast', color = 'forestgreen')
plt.xticks(ticks=range(0,len(df))[::981], labels = df.index[::981])
plt.title('Average AQI Predictions using OLS model')
plt.ylabel('AQI')
plt.xlabel('Time')
plt.grid()
plt.legend(loc='upper right')
plt.show()
#%%
# Hypothesis Testing
# T test
print(f"\nT Test p-values: \n{final_model.pvalues}")
print("As the p-values of the T test is less than the significant level\
 alpha = 0.05, we reject the null hypothesis and conclude that there is a\
 statistically significant relationship between the predictor variable and the response variable.")

# F test
print(f"\nF Test p-value: {final_model.f_pvalue}")
print("As the p-value of the F test is less than the significant level\
 alpha = 0.05, we can reject the null-hypothesis and conclude that final\
 model provides a better fit than the intercept-only model.")


#%%
# AIC, BIC, RMSE, R^2, Adj R^2
print(f"AIC: {final_model.aic:.2f}")
print(f"BIC: {final_model.bic:.2f}")
print(f"RMSE:-")
print(f'\tResidual: {mean_squared_error(Y_train, pred,squared=False):.3f}')
print(f'\tFoercast: {mean_squared_error(Y_test, forecast,squared=False):.3f}')
print(f"R-Squared Value: {(final_model.rsquared*100):.2f}%")
print(f"Adj-R Squared Value: {(final_model.rsquared_adj*100):.2f}%")

print(f"\nOverall the final model's performance is pretty good. In this final model, {(final_model.rsquared_adj*100):.2f}%\
  variation in dependandant variable 'avgAQI' can be explained by the independant variables.\
 The RMSE values are low as well.")
#%%
# ACF of ERRORS
acf(res_err, 50, plot = True, title='ACF of Residual Error')
# acf(fcst_err, 50, plot = True, title='ACF of Forecast Error')

# Model Performance
# MSE
print(f'\nMSE of Residual Error: {mean_squared_error(Y_train, pred):.3f}')
print(f'\nMSE of Foercast Error: {mean_squared_error(Y_test, forecast):.3f}')

# Q
q_res_ols = q_value(res_err, 50, len(Y_train))
print(f'\nQ-Value of Residual Error: {q_res_ols:.3f}')


# Prediction Q
# q_dft = q_value(df_drft['e'],50,len(df_drft))
# print(f"Q-Value: {q_dft:.2f}")

qstar_ols,pvalue_ols=sm.stats.acorr_ljungbox(res_err,lags=[50])

# print(f"Q*-Value of Residual Error: {qstar_ols[0]:.2f}")

# if q_res_ols < qstar_ols:
#   print("The residual is white as Q < Q*")
# else:
#   print("The residual is not white as Q > Q* ")

if pvalue_ols > 0.05:
  print("The residual is white as p > 0.05")
else:
  print("The residual is not white as p < 0.05 ")


# Mean Variance of residual
print(f'Mean of residuals: {np.nanmean(res_err):.2f}')
print(f'Variance of residuals: {np.var(res_err):.2f}')

# %%
######## 13. ARMA, ARIMA, SARIMA
# ARMA
# finding order

# re = acf(Y_train, 50, plot=False)
# Cal_GPAC2(re[50:],7,7)
ACF_PACF_Plot(Y_train, 50)
re = sm.tsa.stattools.acf(Y_train.values, nlags = 50)
Cal_GPAC(re[:],8,8)

print('Observing the patterns ARMA(1,0) and ARMA(3,1) can be selected for farther analysis.')

#%%
# ARMA(1,0)
na = 1
nb = 0
arma10 = sm.tsa.ARMA(Y_train, (na,nb)).fit(trend='nc', disp=0)

# coefficients
for i in range(na):
  print(f"The AR coefficient a{i} is: {-arma10.params[i]:.2f}")
for i in range(nb):
  print(f"The MA coefficient b{i} is {arma10.params[i+na]:.2f}")

print(arma10.summary())

# confidance interval
print('Confidance Interval: ')
print(arma10.conf_int())

print('As the interval does not contain zero in it, it is statistically important.')

# Prediction
arma10_pred = arma10.fittedvalues
arma10_residuals = Y_train - arma10_pred

# Forecast
arma10_for = arma10.predict(start=len(Y_train), end = len(df)-1)
arma10_ferr = pd.DataFrame(Y_test.values - arma10_for).set_index(Y_test.index)
arma10_ferr=pd.Series(np.array(arma10_ferr[0]),index = pd.date_range(Y_test.index[0],periods= len(Y_test)))
# ACF of Residuals
acf(arma10_residuals, 50, plot=True, title= "ACF of ARMA(1,0) Residuals")
acf(arma10_ferr, 50, plot=True, title= "ACF of ARMA(1,0) Forecast Errors")

# MSE
arma10_p_mse = mean_squared_error(Y_train, arma10_pred)
print(f"MSE of Residuals: {arma10_p_mse:.2f}")
arma10_f_mse = mean_squared_error(Y_test, arma10_ferr)
print(f"MSE of Forecast Error: {arma10_f_mse:.2f}")
# Q-Value
arma10_q = q_value(arma10_residuals, 50, len(Y_train))
print(f"Q-Value: {arma10_q:.2f}")

# Covariance Matrix
print('Covariance Matrix: \n', arma10.cov_params())

#%%
# ARMA(3,1)
na = 3
nb = 1
arma31 = sm.tsa.ARMA(Y_train, (na,nb)).fit(trend='nc', disp=0)

# coefficients
for i in range(na):
  print(f"The AR coefficient a{i} is: {-arma31.params[i]:.2f}")
for i in range(nb):
  print(f"The MA coefficient b{i} is {arma31.params[i+na]:.2f}")

print(arma31.summary())

# confidance interval
print('Confidance Interval: ')
print(arma31.conf_int())

print('As the interval does not contain zero in it, it is statistically important.')

# Prediction
arma31_pred = arma31.fittedvalues
arma31_residuals = Y_train - arma31_pred

# Forecast
arma31_for = arma31.predict(start=len(Y_train), end = len(df)-1)
arma31_ferr = pd.DataFrame(Y_test.values - arma31_for).set_index(Y_test.index)
arma31_ferr=pd.Series(np.array(arma31_ferr[0]),index = pd.date_range(Y_test.index[0],periods= len(Y_test)))
# ACF of Residuals
acf(arma31_residuals, 50, plot=True, title= "ACF of ARMA(3,1) Residuals")
acf(arma31_ferr, 50, plot=True, title= "ACF of ARMA(3,1) Forecast Errors")

# MSE
arma31_p_mse = mean_squared_error(Y_train, arma31_pred)
print(f"MSE of Residuals: {arma31_p_mse:.2f}")
arma31_f_mse = mean_squared_error(Y_test, arma31_for)
print(f"MSE of Forecast Error: {arma31_f_mse:.2f}")
# Q-Value
arma31_q = q_value(arma31_residuals, 50, len(Y_train))
print(f"Q-Value: {arma31_q:.2f}")
# Covariance Matrix
print('Covariance Matrix: \n', arma31.cov_params())

print("\nAmong ARMA(1,0) model ARMA(3,1), ARMA(1.0) has lower Q valure but ARMA(3,1) is better at forecasting.")

#%%
# ARIMA
# ARIMA(1,1,0)
na = 1
d = 1
nb = 0
arima110 = sm.tsa.ARIMA(endog=Y_train, order=(na,d,nb)).fit()

# coefficients
for i in range(1,na+1):
  print(f"The AR coefficient a{i} is: {-arima110.params[i]:.2f}")
for i in range(1,nb+1):
  print(f"The MA coefficient b{i} is {arima110.params[i+na]:.2f}")

print(arima110.summary())

# confidance interval
print('Confidance Interval: ')
print(arima110.conf_int())

print('As the interval does not contain zero in it, it is statistically important.')

# Prediction
arima110_pred = arima110.fittedvalues
arima110_predict = inverse_diff(Y_train.values,np.array(arima110_pred),1)
arima110_residuals = Y_train[1:] - arima110_predict

# Forecast
arima110_for = arima110.predict(start=len(Y_train), end = len(df)-1)
arima110_for = inverse_diff(Y_test.values,np.array(arima110_for),1)
arima110_ferr = pd.DataFrame(Y_test.values[:-1] - arima110_for).set_index(Y_test.index[:-1])
arima110_ferr=pd.Series(np.array(arima110_ferr[0]),index = pd.date_range(Y_test.index[0],periods= len(Y_test)-1))

# ACF of Residuals
acf(arima110_residuals, 50, plot=True, title= "ACF of ARIMA(1,1,0) Residuals")
acf(arima110_ferr, 50, plot=True, title= "ACF of ARIMA(1,1,0) Forecast Errors")

# # MSE
arima110_p_mse = mean_squared_error(Y_train[:-1], arima110_predict)
print(f"MSE of Residuals: {arima110_p_mse:.2f}")
arima110_f_mse = mean_squared_error(Y_test[:-1], arima110_for)
print(f"MSE of Forecast Error: {arima110_f_mse:.2f}")
# Q-Value
arima110_q = q_value(arima110_residuals, 50, len(Y_train))
print(f"Q-Value: {arima110_q:.2f}")
# Covariance Matrix
print('Covariance Matrix: \n', arima110.cov_params())

#%%
# ARIMA(3,1,1)
na = 3
d = 1
nb = 1
arima311 = sm.tsa.ARIMA(endog=Y_train, order=(na,d,nb)).fit()

# coefficients
for i in range(1,na+1):
  print(f"The AR coefficient a{i} is: {-arima311.params[i]:.2f}")
for i in range(1,nb+1):
  print(f"The MA coefficient b{i} is {arima311.params[i+na]:.2f}")

print(arima311.summary())

# confidance interval
print('Confidance Interval: ')
print(arima311.conf_int())

print("\nHere interval of AR coefficient a2 contains zero, it is statistically not important in this model.")

# Prediction
arima311_pred = arima311.fittedvalues
arima311_predict = inverse_diff(Y_train.values,np.array(arima311_pred),1)
arima311_residuals = Y_train[1:] - arima311_predict

# Forecast
arima311_for = arima311.predict(start=len(Y_train), end = len(df)-1)
arima311_for = inverse_diff(Y_test.values,np.array(arima311_for),1)
arima311_ferr = pd.DataFrame(Y_test.values[:-1] - arima311_for).set_index(Y_test.index[:-1])
arima311_ferr=pd.Series(np.array(arima311_ferr[0]),index = pd.date_range(Y_test.index[0],periods= len(Y_test)-1))

# ACF of Residuals
acf(arima311_residuals, 50, plot=True, title= "ACF of ARIMA(3,1,1) Residuals")
acf(arima311_ferr, 50, plot=True, title= "ACF of ARIMA(3,1,1) Forecast Errors")

# # MSE
arima311_p_mse = mean_squared_error(Y_train[:-1], arima311_predict)
print(f"MSE of Residuals: {arima311_p_mse:.2f}")
arima311_f_mse = mean_squared_error(Y_test[:-1], arima311_for)
print(f"MSE of Forecast Error: {arima311_f_mse:.2f}")
# Q-Value
arima311_q = q_value(arima311_residuals, 50, len(Y_train))
print(f"Q-Value: {arima311_q:.2f}")
# Covariance Matrix
print('Covariance Matrix: \n', arima311.cov_params())


print("\nAmong ARIMA(1,1,0) model ARMA(3,1,1), ARIMA(3,1,1) has lower Q valure but ARMA(1,1,0) is better at forecasting.")

#%%
# SARIMA

sarima= sm.tsa.statespace.SARIMAX(Y_train,order=(3,0,1),seasonal_order=(0,2,0,7),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
sarima_results=sarima.fit()
print(sarima_results.summary())

# Prediction
sarima_pred_ = sarima_results.get_prediction(start=0, end=len(Y_train), dynamic=False)
sarima_pred = sarima_pred_.predicted_mean

sarima_residuals = Y_train - sarima_pred.values[1:]

# Forecast
sarima_fore = sarima_results.predict(start=0, end =len(Y_test))
sarima_ferr =Y_test - sarima_fore.values[1:]

# ACF
acf(sarima_residuals,50,plot=True,title="ACF of SARIMA Residuals")
acf(sarima_ferr, 50, plot=True, title="ACF of SARIMA Forecast Errors")

# # MSE
sarima_pred_mse = mean_squared_error(Y_train, sarima_pred[1:])
print(f"MSE of Residuals: {sarima_pred_mse:.2f}")
sarima_fore_mse = mean_squared_error(Y_test, sarima_fore[1:])
print(f"MSE of Forecast Error: {sarima_fore_mse:.2f}")
# Q-Value
sarima_q = q_value(sarima_residuals, 50, len(Y_train))
print(f"Q-Value: {sarima_q:.2f}")
# Covariance Matrix
print('Covariance Matrix: \n', sarima_results.cov_params())

# %%
######## 14. LMA

# AR(3) MA(1) 
SSE,cov,teta_hat,var = LMA(Y_train,3,1)

print("Estimated ARMA(3,1) model parameters using the LM Algorithm are:- \n", teta_hat)
print(f"\nStandard deviation of parameter estimates: {np.std(teta_hat):.2f}")
conf_int(cov, teta_hat, 3, 1)
print('\nThe coefficients are statistically important as the interval does not include 0.')

#%%
# coefficents from ARMA(3,1)
print(f"{-arma31.params[:3].values} {arma31.params[-1]}")

# %%
######## 15. Diagnostic Analysis
# confidance intervals
print("####  Confidance Intervals:  ####\n")
print("\nOLS:- \n", final_model.conf_int())
print("\nARMA(1,0):-\n", arma10.conf_int())
print("\nARMA(3,1):-\n", arma31.conf_int())
print("\nARIMA(1,1,0):-\n", arima110.conf_int())
print("\nARIMA(3,1,1):-\n", arima311.conf_int())
print("\nSARIMA:-\n", sarima_results.conf_int())
#%%
# zero/pole cancellation
print("####  Zero/Pole cancellations:  ####\n")
# print("\nOLS:- \n", zero_pole(final_model.params, na))
print("\nARMA(1,0):-\n")
zero_pole(arma10.params, 1)
print("\nARMA(3,1):-\n")
zero_pole(arma31.params, 3)
print("\nARIMA(1,1,0):-\n")
zero_pole(arima110.params, 1)
print("\nARIMA(3,1,1):-\n")
zero_pole(arima311.params, 3)
print("\nSARIMA:-\n")
zero_pole(sarima_results.params[:-1], 3)

print("None of the models have zero pole cancellations.")
#%%
# chi sq test

def chi_sq(lags,na,nb, q, alpha=0.01):
  from scipy.stats import chi2
  DOF= lags - na - nb
  chi_critical = chi2.ppf(1-alpha,DOF)
  print(f"\tQ-Value: {q:.2f}\n\tChi Critical Value: {chi_critical:.2f}")
  if q < chi_critical:
    print('The residual is white')
  else:
      print('The residual is not white')
  return None


print("\nARMA(1,0):-")
chi_sq(50,1,0,arma10_q)
print("\nARMA(3,1):-")
chi_sq(50,3,1,arma31_q)
print("\nARIMA(1,1,0):-\n")
chi_sq(50,1,0,arima110_q)
print("\nARIMA(3,1,1):-\n")
chi_sq(50,3,0,arima311_q)
print("\nSARIMA:-\n")
chi_sq(50,3,0,sarima_q)

#%%
# Q-Values
print("Q-Values of Residual Error:")
print(f'\tOLS: {q_res_ols:.3f}')
print(f'\tARMA(1,0): {arma10_q:.3f}')
print(f'\tARMA(3,1): {arma31_q:.3f}')
print(f'\tARIMA(1,1,0): {arima110_q:.3f}')
print(f'\tARIMA(3,1,1): {arima311_q:.3f}')
print(f'\tSARIMA(3,0,1) x (0,2,0,7) : {sarima_q:.3f}')

#%%
# variance of residual error
print("\nVariance of Residual Errors: ")
print(f"\tOLS: {np.var(res_err):.2f}")
print(f"\tARMA(1,0): {np.var(arma10_residuals):.2f} ")

print(f"\tARMA(3,1): {np.var(arma31_residuals):.2f} ")

print(f"\tARIMA(1,1,0): {np.var(arima110_residuals):.2f} ")

print(f"\tARIMA(3,1,1): {np.var(arima311_residuals):.2f} ")

print(f"\tSARIMA: {np.var(sarima_residuals):.2f} ")

# variance of forecast error
print("\nVariance of Forecast Errors: ")
print(f"\tOLS: {np.var(fcst_err):.2f}")
print(f"\tARMA(1,0): {np.var(arma10_ferr):.2f} ")

print(f"\tARMA(3,1): {np.var(arma31_ferr):.2f} ")

print(f"\tARIMA(1,1,0): {np.var(arima110_ferr):.2f} ")

print(f"\tARIMA(3,1,1): {np.var(arima311_ferr):.2f} ")

print(f"\tSARIMA: {np.var(sarima_ferr):.2f} ")

# MSE
# MSE of residuals
print("\nMSE of Residuals: ")
print(f"\tOLS: {np.mean(np.square(res_err)):.2f}")
print(f"\tARMA(1,0): {np.mean(np.square(arma10_residuals)):.2f} ")

print(f"\tARMA(3,1): {np.mean(np.square(arma31_residuals)):.2f} ")

print(f"\tARIMA(1,1,0): {np.mean(np.square(arima110_residuals)):.2f} ")

print(f"\tARIMA(3,1,1): {np.mean(np.square(arima311_residuals)):.2f} ")

print(f"\tSARIMA: {np.mean(np.square(sarima_residuals)):.2f} ")

# MSE of forecasts
print("\nMSE of Forecasts: ")
print(f"\tOLS: {np.mean(np.square(fcst_err)):.2f}")
print(f"\tARMA(1,0): {np.mean(np.square(arma10_ferr)):.2f} ")

print(f"\tARMA(3,1): {np.mean(np.square(arma31_ferr)):.2f} ")

print(f"\tARIMA(1,1,0): {np.nanmean(np.square(arima110_ferr)):.2f} ")

print(f"\tARIMA(3,1,1): {np.mean(np.square(arima311_ferr)):.2f} ")

print(f"\tSARIMA: {np.mean(np.square(sarima_ferr)):.2f} ")

#%%

# ratio of test set by forecast
print("\nRatio of test set variance by forecast variance: ")
print(f"\tOLS: {np.var(Y_test)/np.var(forecast):.2f}")
print(f"\tARMA(1,0): {np.var(Y_test)/np.var(arma10_for):.2f} ")

print(f"\tARMA(3,1): {np.var(Y_test)/np.var(arma31_for):.2f} ")

print(f"\tARIMA(1,1,0): {np.var(Y_test)/np.var(arima110_for):.2f} ")

print(f"\tARIMA(3,1,1): {np.var(Y_test)/np.var(arima311_for):.2f} ")

print(f"\tSARIMA: {np.var(Y_test)/np.var(sarima_fore):.2f} ")


# %%
######## 17. Final model Selection

print("The final model is ARIMA(3,1,1)")

# %%
######## 18. Forecast Function

y_train_diff = y_train.diff(1).dropna()
y_hat = []
for i in range(1,len(y_train_diff)):
    if i==1:
      y_hat.append((0.69*y_train_diff[i-1]) -(0.97 *y_train_diff[i-1]))
    elif i == 2:
      y_hat.append((0.69*y_train_diff[i-1]) - (0.09 * y_train_diff[i-2]) - (0.97*(y_train_diff[i-1] - y_hat[0])))
    else:
      y_hat.append((0.69*y_train_diff[i-1]) - (0.09*y_train_diff[i-2]) -(0.01*y_train_diff[i-3]) - (0.97*(y_train_diff[i-1] - y_hat[-1])) )

y_hat_inv_diff = inverse_diff(y_train.values,np.array(y_hat),1)

#%%
plt.plot(y_train,label='True Data (Train set)')
plt.plot(y_hat_inv_diff,label='Fitted Data (1-step prediction)')
plt.title('True data vs. One step prediction data')
plt.xticks(ticks=range(0,len(y_train_diff))[::697], labels = y_train_diff.index[::697], rotation = 90)
plt.suptitle("ARIMA(3,1,1): y(t) – 0.69 y(t-1) + 0.09 y(t-2) + 0.01 y(t-3) = e(t) – 0.97 e(t-1)", fontsize=22)
plt.legend(loc='upper right', bbox_to_anchor=(1.01,1))
plt.xlabel('Time')
plt.ylabel('Average AQI')
plt.show()


# %%
######## 19. h-step Prediction
def h_step(h,y_train, y_test, y):
  
  y_hat = []
  for i in range(len(y_train),len(y)):
    if i==len(y_train):
      
      y_hat.append((0.69*y[i-h]) -(0.97 *y[i-h]))
    elif i == (len(y_train)+1):
      
      y_hat.append((0.69*y[i-h]) - (0.09 * y[i-h-1]) - (0.97*(y[i-h] - y_hat[0])))
    else:
      
      y_hat.append((0.69*y[i-h]) - (0.09*y[i-h-1]) -(0.01*y[i-h-2]) - (0.97*(y[i-h] - y_hat[-1])) )
  return y_hat

h=30
arima311_hstep = h_step(h,y_train, y_test,df['avgAQI'].diff(1).dropna())

arima311_hstep_inv_diff = inverse_diff(y_test.values,np.array(arima311_hstep),1)

plt.plot(y_test,label='True Data (Test set)')
plt.plot(arima311_hstep_inv_diff,label='Fitted Data (h-step prediction)')
plt.title(f'True data vs. {h}-step prediction data')
plt.xticks(ticks=range(0,len(y_test))[::174], labels = y_test.index[::174], rotation = 90)
plt.suptitle("ARIMA(3,1,1): y(t) – 0.69 y(t-1) + 0.09 y(t-2) + 0.01 y(t-3) = e(t) – 0.97 e(t-1)", fontsize=22)
plt.legend(loc='upper right', bbox_to_anchor=(1.01,1))
plt.xlabel('Time')
plt.ylabel('Average AQI')
plt.show()

#%%
# variance of test set vs vaarience of predicted set
print(f"Variance of test set: {np.var(y_test):.2f}")
print(f"Variance of predicted set: {np.var(arima311_hstep_inv_diff):.2f}")
print(f"Ratio: {np.var(y_test)/np.var(arima311_hstep_inv_diff):.2f}")

# %%
######## 20. Summary  and Conclusion

# Among the ARMA and ARIMA models ARIMA model with AR (3), MA (1) with differencing order 1,
#  performed better than the other by considering the lowest q-value of residuals, MSE, ratio
#  of variance of test set vs forecasted set. From this model I farther generated the model
#  equation and built 1-step and multi-step prediction functions and the model is exceptionally
#  good at predict next week as well as next month’s average AQI. Overall, the models did not
#  have white q-value of residuals possibly because of the nature of the dataset. More advanced
#  machine learning technique of forecasting like LSTM, XGboost etc. can possibly achieve that.

# %%
