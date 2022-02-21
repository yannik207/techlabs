import pandas as pd

from flood_class import flood

 
prediction = flood()
prediction.data = pd.read_csv('Bangladesh.csv', sep=';', encoding= 'unicode_escape')
prediction.place = 'Bhola'
prediction.feature = ['Rainfall'] #'Cloud_Coverage','Bright_Sunshine','Wind_Speed','Max_Temp','Min_Temp','Relative_Humidity','Sl','Year','Month', 'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT', 'Period'

prediction.preprocess()     # clear Dataset
prediction.split()          #split into test ans train
prediction.LR_all()         #Linear Regression train all stations, test all stations
prediction.LR_all_one()     #Linear Regression train all stations, test one station
prediction.LR_one()         #Linear Regression train one station, test one station
prediction.LR_one_all()     #Linear Regression train one station, test all stations