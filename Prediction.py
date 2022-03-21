import pandas as pd

from flood_class2 import flood

 
prediction = flood()

prediction.data = pd.read_csv('Bangladesh.csv', sep=';', encoding= 'unicode_escape')
prediction.place = 'Bhola' #when all=False use this station
prediction.feature = ['Rainfall','Cloud_Coverage','Bright_Sunshine','Wind_Speed','Max_Temp','Min_Temp','Relative_Humidity'] #'Cloud_Coverage','Bright_Sunshine','Wind_Speed','Max_Temp','Min_Temp','Relative_Humidity','Sl','Year','Month', 'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT', 'Period'
prediction.test_size = 0.3

prediction.preprocess()     #clear Dataset
prediction.split()          #split into test ans train

#prediction.LR_train(all=True)         #Logistic Regression train all stations, test all stations
#prediction.LR_prediction(all=True)

prediction.Random_Forest()

'''
# Predicting second dataset without labels with trained model
prediction.data = pd.read_csv() #prediction dataset
prediction.new()
'''
