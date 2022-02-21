import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix


class flood(object):
	def __init__(self,**kwargs): 
		self.data=kwargs.get("data", None)
		self.feature=kwargs.get("feature", None)
		self.place=kwargs.get("place", None)
		self.test_size=kwargs.get("test_size", 0.8)


	def preprocess(self):
        #ceep only relevant columns in the dataset 
		self.data = self.data[['Station_Names']+self.feature+['Flood?']]

        #replace , in dataset
		for i in self.feature:
			self.data[i]=self.data[i].str.replace(',','.')

        #Cheak if any colomns are left empty
		self.data.apply(lambda x:sum(x.isnull()), axis=0)

        #We want the data in numbers, therefore we will replace the yes/no in floods coloumn by 1/0
		self.data.fillna(0,inplace=True)

		names = ['Barisal','Bhola','Bogra','Chandpur','Comilla',"Cox's Bazar",'Dhaka','Dinajpur','Faridpur','Feni','Hatiya','Ishurdi','Jessore']

        #Devide dataset in individual regions
        #Devide dataset in training data(data_stations[name][0]) and label(data_stations[name][1])

		self.data_stations = {}

		groups = self.data.groupby(self.data.Station_Names)

		#for i in names:
		#	stations = (groups.get_group(i))
		#	self.data_stations[i] = [stations.iloc[:,1:-1], stations.iloc[:,-1]]

		stations = (groups.get_group(self.place))
		self.data_stations[self.place] = [stations.iloc[:,1:-1], stations.iloc[:,-1]]

		self.data = self.data.drop(columns=['Station_Names'])
		self.x=self.data.iloc[:,:-1]
		self.y=self.data.iloc[:,-1]


	def split(self):

		self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=self.test_size)
		self.x_train2,self.x_test2,self.y_train2,self.y_test2=train_test_split(self.data_stations[self.place][0],self.data_stations[self.place][1],test_size=self.test_size)
    

        # type casting - changing the datatype from float to integer.
		self.y_train=self.y_train.astype('int')
		self.y_test=self.y_test.astype('int')

		self.y_train2=self.y_train2.astype('int')
		self.y_test2=self.y_test2.astype('int')
        
	def feature_importance(self,lr):
        
		importance = lr.coef_[0]
		for i,v in enumerate(importance):
			print('Feature {}: {}, Score: {}'.format(i,self.feature[i],np.round(v,2)))
    
        # plot feature importance
		plt.bar([x for x in range(len(importance))], importance)
		plt.show()


	def LR_all(self):
		# Scaling the data between 0 and 1.
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.x).transform(self.x)
        
		x_train_std=minmax.fit_transform(self.x_train)         # fit the values in between 0 and 1.
		x_test_std=minmax.fit_transform(self.x_test)

		lr=LogisticRegression()
		lr.fit(x_train_std,self.y_train)	 # train the model based on full dataset x&y

		self.feature_importance(lr)

		lr_acc=cross_val_score(lr,x_train_std,self.y_train,cv=3,scoring='accuracy',n_jobs=-1) #cross valuation of logisitc regression accuracy (Accuracy is the proportion of correct predictions over total predictions.)
		lr_proba=cross_val_predict(lr,x_train_std,self.y_train,cv=3,method='predict_proba') #cross valuation logistic regression probability (Probability is the chance for class 1 (flood) to occure)
     							
		y_pred=lr.predict(x_test_std)
        
		print("train all stations, test all stations")
		print("\naccuracy score:%f"%(accuracy_score(self.y_test,y_pred)*100))
		print("recall score:%f"%(recall_score(self.y_test,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(self.y_test,y_pred)*100))
		print(confusion_matrix(self.y_test,y_pred))

	def LR_all_one(self):
        # Scaling the data between 0 and 1.
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.x).transform(self.x)
        

		x_train_std=minmax.fit_transform(self.x_train)         # fit the values in between 0 and 1.
		x_test_std=minmax.fit_transform(self.x_test2)

		lr=LogisticRegression()
		lr.fit(x_train_std,self.y_train)	 # train the model based on full dataset x&y

		self.feature_importance(lr)

		lr_acc=cross_val_score(lr,x_train_std,self.y_train,cv=3,scoring='accuracy',n_jobs=-1) #cross valuation of logisitc regression accuracy (Accuracy is the proportion of correct predictions over total predictions.)
		lr_proba=cross_val_predict(lr,x_train_std,self.y_train,cv=3,method='predict_proba') #cross valuation logistic regression probability (Probability is the chance for class 1 (flood) to occure)
     							
		y_pred=lr.predict(x_test_std)
        
		print("train all stations, test one stations")
		print("\naccuracy score:%f"%(accuracy_score(self.y_test2,y_pred)*100))
		print("recall score:%f"%(recall_score(self.y_test2,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(self.y_test2,y_pred)*100))
		print(confusion_matrix(self.y_test2,y_pred))

	def LR_one(self):
        # Scaling the data between 0 and 1.
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.data_stations[self.place][0]).transform(self.data_stations[self.place][0])
        

		x_train_std=minmax.fit_transform(self.x_train2)         # fit the values in between 0 and 1.
		x_test_std=minmax.fit_transform(self.x_test2)

		lr=LogisticRegression()
		lr.fit(x_train_std,self.y_train2)	 # train the model based on one station

		self.feature_importance(lr)

		lr_acc=cross_val_score(lr,x_train_std,self.y_train2,cv=3,scoring='accuracy',n_jobs=-1) #cross valuation of logisitc regression accuracy (Accuracy is the proportion of correct predictions over total predictions.)
		lr_proba=cross_val_predict(lr,x_train_std,self.y_train2,cv=3,method='predict_proba') #cross valuation logistic regression probability (Probability is the chance for class 1 (flood) to occure)
     							
		y_pred=lr.predict(x_test_std)
        
		print("train one station, test one station")
		print("\naccuracy score:%f"%(accuracy_score(self.y_test2,y_pred)*100))
		print("recall score:%f"%(recall_score(self.y_test2,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(self.y_test2,y_pred)*100))
		print(confusion_matrix(self.y_test2,y_pred))


	def LR_one_all(self):
        # Scaling the data between 0 and 1.
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.data_stations[self.place][0]).transform(self.data_stations[self.place][0])
        

		x_train_std=minmax.fit_transform(self.x_train2)         # fit the values in between 0 and 1.
		x_test_std=minmax.fit_transform(self.x_test)

		lr=LogisticRegression()
		lr.fit(x_train_std,self.y_train2)	 # train the model based on one station

		self.feature_importance(lr)

		lr_acc=cross_val_score(lr,x_train_std,self.y_train2,cv=3,scoring='accuracy',n_jobs=-1) #cross valuation of logisitc regression accuracy (Accuracy is the proportion of correct predictions over total predictions.)
		lr_proba=cross_val_predict(lr,x_train_std,self.y_train2,cv=3,method='predict_proba') #cross valuation logistic regression probability (Probability is the chance for class 1 (flood) to occure)
     							
		y_pred=lr.predict(x_test_std)
        
		print("train one station, test all stations")
		print("\naccuracy score:%f"%(accuracy_score(self.y_test,y_pred)*100))
		print("recall score:%f"%(recall_score(self.y_test,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(self.y_test,y_pred)*100))
		print(confusion_matrix(self.y_test,y_pred))
