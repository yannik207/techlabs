# ============================================================
# file:              https://github.com/n-gauhar/Flood-prediction
# Author:            Yannik Sacherer
# necessary packages: Pandas, Numpy, Datatile
# ============================================================


###load packages --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datatile.summary.df import DataFrameSummary
###load data --------------------------------------
df = pd.read_csv('/Users/yanniksa/Desktop/bangladesh.csv', sep=';')

###Data Overview --------------------------------------

df_summary = DataFrameSummary(df)
print(df_summary.summary())
print(df)
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
df['Min_Temp'] = df['Min_Temp'].str.extract('(\d+)')
df['Cloud_Coverage'] = df['Cloud_Coverage'].str.extract('(\d+)')
print(df.info())
df['Cloud_Coverage'] = df['Cloud_Coverage'].astype('int64')
df['Max_Temp'] = df['Max_Temp'].astype('int64')
df['Min_Temp'] = df['Min_Temp'].astype('int64')
df['Rainfall'] = df['Rainfall'].str.extract('(\d+)')
df['Rainfall'] = df['Rainfall'].astype('int64')
df['Relative_Humidity'] = df['Relative_Humidity'].str.extract('(\d+)')
df['Relative_Humidity'] = df['Relative_Humidity'].astype('int64')

del df['Bright_Sunshine']





###explorative analysis --------------------------------------

num_variable = ['Sl', 'Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
                'Relative_Humidity', 'Cloud_Coverage', 'Station_Number', 'X_COR', 'Y_COR', 'LONGITUDE', 'ALT', 'Period', 'Flood?']
for i in num_variable:
    print(i)


for i in num_variable:
    plt.figure()
    plt.hist(df[i])
    plt.title(i)
    plt.show()
###Data preperation --------------------------------------

###Modelling --------------------------------------

###Evaluation --------------------------------------