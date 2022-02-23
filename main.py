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
print(df.info())
df.isnull().values.any()

# Latitude has wrong values. change it with domain knowledge
# Observation 0 to 1355 has the value of '22. Juli' instead of 22.17

print(df.head(5))


# remove string from temp observations
header = df[['Max_Temp', 'Min_Temp', 'Cloud_Coverage', 'Rainfall', 'Relative_Humidity']]



df['LATITUDE'] = df['LATITUDE'].apply(lambda x: x.replace(',', '.'))
df['LATITUDE'] = df['LATITUDE'].astype('float64')

df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: x.replace(',', '.'))
df['LONGITUDE'] = df['LONGITUDE'].astype('float64')

df['Period'] = df['Period'].apply(lambda x: x.replace(',', '.'))
df['Period'] = df['Period'].astype('float64')
print(df.info())
#del df['Bright_Sunshine']





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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[['ALT', 'Bright_Sunshine', 'Cloud_Coverage', 'LATITUDE', 'LONGITUDE', 'Max_Temp', 'Min_Temp', 'Month', 'Period', 'Rainfall', 'Relative_Humidity', 'Sl', 'X_COR', 'Y_COR']]

# performin standardization
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

components = None
pca = PCA(n_components = 0.85)
# perform PCA on the scaled data
pca.fit(X_scaled)

