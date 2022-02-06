# ============================================================
# file:              https://github.com/n-gauhar/Flood-prediction
# Author:            Yannik Sacherer
# necessary packages: Pandas, Numpy, Datatile
# ============================================================


###load packages --------------------------
import pandas as pd
import numpy as np
from datatile.summary.df import DataFrameSummary
###load data --------------------------------------
df = pd.read_csv('/Users/yanniksa/Desktop/bangladesh.csv', sep=';')

###Data Overview --------------------------------------

df_summary = DataFrameSummary(df)
print(df_summary.summary())

###Data Cleaning ---------------------------------------
# to handle missing values in the column of 'Flood?'. we fill in 0
df['Flood?'] = df['Flood?'].fillna(0)
df.isnull().values.any()

# Latitude has wrong values. change it with domain knowledge
# Observation 0 to 1355 has the value of '22. Juli' instead of 22.17
df['LATITUDE'] = df['LATITUDE'].replace('22. Jul', 22.17)



###explorative analysis --------------------------------------




###Data preperation --------------------------------------

###Modelling --------------------------------------

###Evaluation --------------------------------------