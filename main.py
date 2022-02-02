# ============================================================
# file:              https://github.com/n-gauhar/Flood-prediction
# Author:            Yannik Sacherer
# necessary packages: Pandas, Numpy, Datatile
# ============================================================


#load packages --------------------------
import pandas as pd
import numpy as np
from datatile.summary.df import DataFrameSummary
#load data --------------------------------------
df = pd.read_csv('/Users/yanniksa/Desktop/bangladesh.csv', sep=';')

#Data Overview --------------------------------------
print(df.head(5))
df_summary = DataFrameSummary(df)
print(df_summary.summary())
#explorative analysis --------------------------------------

#Data preperation --------------------------------------

#Modelling --------------------------------------

#Data Understanding --------------------------------------

#Evaluation --------------------------------------