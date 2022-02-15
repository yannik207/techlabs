

###load packages --------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datatile.summary.df import DataFrameSummary
###load data --------------------------------------
df = pd.read_csv('/Users/Real/Desktop/J.A.R.V.I.S./Techlabs/Bangladesh-2.csv', sep=';')

###Data Overview --------------------------------------

#df_summary = DataFrameSummary(df)
#print(df_summary.summary())
#print(df)
###Data Cleaning ---------------------------------------
# to handle missing values in the column of 'Flood?'. we fill in 0
df['Flood?'] = df['Flood?'].fillna(0)
df.isnull().values.any()
print(df.columns)
df['Max_Temp'] = df['Max_Temp'].str.extract('(\d+)')
df['Min_Temp'] = df['Min_Temp'].str.extract('(\d+)')
df['Cloud_Coverage'] = df['Cloud_Coverage'].str.extract('(\d+)')
df['Cloud_Coverage'] = df['Cloud_Coverage'].astype('int64')
df['Max_Temp'] = df['Max_Temp'].astype('int64')
df['Min_Temp'] = df['Min_Temp'].astype('int64')
df['Rainfall'] = df['Rainfall'].str.extract('(\d+)')
df['Rainfall'] = df['Rainfall'].astype('int64')
df['Relative_Humidity'] = df['Relative_Humidity'].str.extract('(\d+)')
df['Relative_Humidity'] = df['Relative_Humidity'].astype('int64')

#FC: Flooded cities ;  RC = Rainy Cities ; Sunshine Cities
FC = df.groupby("Station_Names")["Flood?"].sum()
print(FC.sort_values(ascending = False))

RC = df.groupby("Station_Names")["Rainfall"].sum()
print(RC.sort_values(ascending = False))

SC = df.groupby("Station_Names")["Bright_Sunshine"].sum()
print(SC.sort_values(ascending = False))

MaxTemperature = df.groupby("Max_Temp")["Flood?"].sum()
print(MaxTemperature.sort_values(ascending = False))

MinTemperature = df.groupby("Min_Temp")["Flood?"].sum()
print(MinTemperature.sort_values(ascending = False))  #Besonders heiße Gebiete haben sehr häufig Fluten - wer hätte es gedacht

#Correlation?  Also: equal entries for each city? roughly





