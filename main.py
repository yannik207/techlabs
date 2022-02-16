# ============================================================
# file:              https://github.com/n-gauhar/Flood-prediction
# Author:            Yannik Sacherer
# necessary packages: Pandas, Numpy, Datatile
# ============================================================


###load packages --------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datatile.summary.df import DataFrameSummary
###load data --------------------------------------
df = pd.read_csv('/Users/yanniksa/Desktop/bangladesh.csv', sep=';')

###Data Overview --------------------------------------

df_summary = DataFrameSummary(df)
print(df_summary.summary())
print(df)
print(df.info())
###Data Cleaning ---------------------------------------
# to handle missing values in the column of 'Flood?'. we fill in 0
df['Flood?'] = df['Flood?'].fillna(0)
df.isnull().values.any()

# Latitude has wrong values. change it with domain knowledge
# Observation 0 to 1355 has the value of '22. Juli' instead of 22.17

df['LATITUDE'] = df['LATITUDE'].replace('22. Jul', 22.17)
print(df.head(5))


# remove string from temp observations
df['Max_Temp'] = df['Max_Temp'].str.extract('(\d+)')
df['Max_Temp'] = df['Max_Temp'].astype('int64')
df['Min_Temp'] = df['Min_Temp'].str.extract('(\d+)')
df['Min_Temp'] = df['Min_Temp'].astype('int64')
df['Cloud_Coverage'] = df['Cloud_Coverage'].str.extract('(\d+)')
df['Cloud_Coverage'] = df['Cloud_Coverage'].astype('int64')
df['Rainfall'] = df['Rainfall'].str.extract('(\d+)')
df['Rainfall'] = df['Rainfall'].astype('int64')
df['Relative_Humidity'] = df['Relative_Humidity'].str.extract('(\d+)')
df['Relative_Humidity'] = df['Relative_Humidity'].astype('int64')

del df['Bright_Sunshine']





###explorative analysis --------------------------------------

num_variable = ['Sl', 'Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
                'Relative_Humidity', 'Cloud_Coverage', 'Station_Number', 'X_COR', 'Y_COR', 'LONGITUDE', 'ALT', 'Period', 'Flood?']

for i in num_variable:
    plt.figure()
    plt.hist(df[i])
    plt.title(i)
    #plt.show()

num_variable = df[['Sl', 'Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
                'Relative_Humidity', 'Cloud_Coverage', 'Station_Number', 'X_COR', 'Y_COR', 'LONGITUDE', 'ALT', 'Period']]

print(round(num_variable.corr(), 2))

corr = num_variable.corr()
plt.figure()
ax = sns.clustermap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    square=True
)
plt.show()

#FC: Flooded cities ;  RC = Rainy Cities ; Sunshine Cities
FC = df.groupby("Station_Names")["Flood?"].sum()
print(FC.sort_values(ascending = False))

RC = df.groupby("Station_Names")["Rainfall"].sum()
print(RC.sort_values(ascending = False))

#SC = df.groupby("Station_Names")["Bright_Sunshine"].sum()
#print(SC.sort_values(ascending = False))

MaxTemperature = df.groupby("Max_Temp")["Flood?"].sum()
print(MaxTemperature.sort_values(ascending = False))

MinTemperature = df.groupby("Min_Temp")["Flood?"].sum()
print(MinTemperature.sort_values(ascending = False))  #Besonders heiße Gebiete haben sehr häufig Fluten - wer hätte es gedacht

###Data preperation --------------------------------------

###Modelling --------------------------------------

###Evaluation --------------------------------------