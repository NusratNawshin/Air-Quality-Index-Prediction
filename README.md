# Air-Quality-Index-Prediction
Time Series Analysis on California State Loss Angeles County

# DATS6313 TIME SERIES ANALYSIS & MODELLING
#### With Prof. Reza Jafari

### George Washington University
**Spring 2022**

#
The project is to analyze and predict Air Quality Index using different type of simple forecasting method, Multiple Regression model, ARMA, ARIMA and SARIMA models. The dataset US Air Pollution 2000-2021 is collected from Kaggle. Link of the dataset is [https://www.kaggle.com/datasets/alpacanonymous/us-pollution-20002021]

#
The final dataset contains 7,852 number of rows and 24 columns. 
Numeric columns list:
- Year
- Month
- Day
- O3 Mean
- O3 1st Max Value
- O3 1st Max Hour 
- O3 AQI
- CO Mean
- CO 1st Max Value
- CO 1st Max Hour
- CO AQI 
- SO2 Mean
- SO2 1st Max Value
- SO2 1st Max Hour
- SO2 AQI 
- NO2 Mean
- NO2 1st Max Value
- NO2 1st Max Hour 
- NO2 AQI
- 23  avgAQI 

Categorical column list:
- Address
- State 
- County
- City

#
In order to run the project these packages need to be pre-installed into the python environment.

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- sklearn
- scipy
- math

**The command to install all packages together is:

pip install pandas numpy matplotlib seaborn statsmodels sklearn scipy math**

#### dataset:
The datasets has to be stored inside data folder.

**pollution_2000_2021.csv** : US all state air pollution data

**AQI_CA_LA.csv** : Only California state Loss Angeles County air pollution data


#### files:
**data_preparation.py** - Reads the original dataset 'pollution_2000_2021.csv' and generates final dataset for analysis 'AQI_CA_LA.csv'
**toolbox.py** - Contains all the helper functions for the analysis
**analysis.py** - Reads the 'AQI_CA_LA.csv' and contains all the time series analysis
**NN_TSA_FTP_Report.pdf** - Final project report pdf

