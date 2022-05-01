In order to run the project these packages need to be pre-installed into the python environment.

pandas
numpy
matplotlib
seaborn
statsmodels
sklearn
scipy
math

**The command to install all packages together is:

pip install pandas numpy matplotlib seaborn statsmodels sklearn scipy math**

dataset:
The datasets has to be stored inside data folder.

pollution_2000_2021.csv : US all state air pollution data

AQI_CA_LA.csv : Only California state Loss Angeles County air pollution data

files:
data_preparation.py - Reads the original dataset 'pollution_2000_2021.csv' and generates final dataset for analysis 'AQI_CA_LA.csv'

toolbox.py - Contains all the helper functions for the analysis

analysis.py - Reads the 'AQI_CA_LA.csv' and contains all the time series analysis