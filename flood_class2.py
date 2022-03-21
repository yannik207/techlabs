import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import loguniform
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class flood(object):
	def __init__(self,**kwargs): 
		self.data=kwargs.get("data", None)
		self.feature=kwargs.get("feature", ['Rainfall'])
		self.place=kwargs.get("place", None)
		self.test_size=kwargs.get("test_size", 0.8)


	def preprocess(self):
        #ceep only relevant columns in the dataset 
		self.data = self.data.sort_values(by=['Period'])

		self.data = self.data[['Station_Names']+self.feature+['Flood?']]

        #replace , in dataset
		for i in self.feature:
			self.data[i]=self.data[i].str.replace(',','.')

        #Cheak if any colomns are left empty
		self.data.apply(lambda x:sum(x.isnull()), axis=0)

        #We want the data in numbers, therefore we will replace the yes/no in floods coloumn by 1/0
		self.data.fillna(0,inplace=True)

		#names = ['Barisal','Bhola','Bogra','Chandpur','Comilla',"Cox's Bazar",'Dhaka','Dinajpur','Faridpur','Feni','Hatiya','Ishurdi','Jessore']

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
		len_train = len(self.x_train)
		self.x_train = self.x.iloc[:len_train]
		self.x_test = self.x.iloc[len_train:]
		self.y_train = self.y.iloc[:len_train]
		self.y_test = self.y.iloc[len_train:]

		self.x_train2,self.x_test2,self.y_train2,self.y_test2=train_test_split(self.data_stations[self.place][0],self.data_stations[self.place][1],test_size=self.test_size)
		len_train = len(self.x_train2)
		self.x_train2 = self.data_stations[self.place][0].iloc[:len_train]
		self.x_test2 = self.data_stations[self.place][0].iloc[len_train:]
		self.y_train2 = self.data_stations[self.place][1].iloc[:len_train]
		self.y_test2 = self.data_stations[self.place][1].iloc[len_train:]


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
		LR_importances = pd.Series(importance, index=self.feature)

		fig, ax = plt.subplots()
		LR_importances.plot.bar( ax=ax)
		ax.set_title("Feature importances")
		ax.set_ylabel("Importance")
		fig.tight_layout()
		plt.show()


	def LR_train(self,all):
		# Scaling the data between 0 and 1.
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		if all == True:
			minmax.fit(self.x).transform(self.x)
			x_train = self.x_train2
			y_train = self.y_train2
		if all == False:
			minmax.fit(self.data_stations[self.place][0]).transform(self.data_stations[self.place][0])
			x_train = self.x_train
			y_train = self.y_train
		
        
		x_train_std=minmax.fit_transform(x_train)         # fit the values in between 0 and 1.


		lr=LogisticRegression()
		my_cv = TimeSeriesSplit(n_splits=10).split(x_train_std)
		space = dict()
		space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
		space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
		space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
		search = GridSearchCV(lr, space, scoring='roc_auc', n_jobs=-1,cv=my_cv)

		self.result = search.fit(x_train_std,y_train)	 # train the model based on full dataset x&y

		print('Best Score: %s' % self.result.best_score_)
		print('Best Hyperparameters: %s' % self.result.best_params_)

		self.feature_importance(self.result.best_estimator_)

		filename = 'logistic_regression_model.sav'
		pickle.dump(self.result.best_estimator_, open(filename, 'wb'))

		if all == True:
			print("train all stations")
		if all == False:
			print("train one station")
		
		
	def LR_prediction(self, all):
		try:
			filename = 'logistic_regression_model.sav'
			model = pickle.load(open(filename, 'rb'))
		except:
			model = self.result.best_estimator_

		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.x).transform(self.x)

		if all == False:
			x_test = self.x_test2
			y_test = self.y_test2

		if all == True:
			x_test = self.x_test
			y_test = self.y_test

		x_test_std=minmax.fit_transform(x_test)
		y_pred=model.predict(x_test_std)
        
		if all == True:
			print("test all stations")
		if all == False:
			print("test one station")
		print("\naccuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
		print("recall score:%f"%(recall_score(y_test,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(y_test,y_pred)*100))
		print(confusion_matrix(y_test,y_pred))

	def Random_Forest(self):
		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.x).transform(self.x)

		x_train_std=minmax.fit_transform(self.x_train)         # fit the values in between 0 and 1.
		x_test_std=minmax.fit_transform(self.x_test)

		rmf=RandomForestClassifier(max_depth=3,random_state=0)

		# Number of trees in random forest
		n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
		# Number of features to consider at every split
		max_features = ['auto', 'sqrt']
		# Maximum number of levels in tree
		max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
		max_depth.append(None)
		# Minimum number of samples required to split a node
		min_samples_split = [2, 5, 10]
		# Minimum number of samples required at each leaf node
		min_samples_leaf = [1, 2, 4]
		# Method of selecting samples for training each tree
		bootstrap = [True, False]

		# Create the random grid
		random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

		'''
		rmf_random = RandomizedSearchCV(estimator=rmf, param_distributions=random_grid,
                              n_iter = 100, scoring='roc_auc', 
                              cv = TimeSeriesSplit(n_splits=3), verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)
		'''
		print('start Random forrest')
		my_cv = TimeSeriesSplit(n_splits=3).split(x_train_std)
		print('startGridSearch')
		rmf_random = GridSearchCV(estimator=rmf, param_grid=random_grid, cv=my_cv)
		print('start fit')
		rmf_clf=rmf_random.fit(x_train_std,self.y_train)
		print('finished')
		print(rmf_random.best_params_)

		#rmf_clf_acc=cross_val_score(rmf_clf,x_train_std,self.y_train,cv=3,scoring="accuracy",n_jobs=-1)
		#rmf_proba=cross_val_predict(rmf_clf,x_train_std,self.y_train,cv=3,method='predict_proba')

		y_pred=rmf_clf.best_estimator_.predict(x_test_std)

		importances = rmf_clf.best_estimator_.feature_importances_ #Feature importances are provided by the fitted attribute feature_importances_ and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.
		#std = np.std([tree.feature_importances_ for tree in rmf.estimators_], axis=0)

		forest_importances = pd.Series(importances, index=self.feature)

		fig, ax = plt.subplots()
		forest_importances.plot.bar(ax=ax)
		ax.set_title("Feature importances using MDI")
		ax.set_ylabel("Mean decrease in impurity")
		fig.tight_layout()
		plt.show()

		#print("\naccuracy score:%f"%(accuracy_score(self.y_test,y_pred)*100))
		#print("recall score:%f"%(recall_score(self.y_test,y_pred)*100))
		print("roc score:%f"%(roc_auc_score(self.y_test,y_pred)*100))
		print(confusion_matrix(self.y_test,y_pred))

	def new(self):
		
        #replace , in dataset
		for i in self.feature:
			self.data[i]=self.data[i].str.replace(',','.')

		self.x = self.data[self.feature]

        #Cheak if any colomns are left empty
		self.x.apply(lambda x:sum(x.isnull()), axis=0)

		try:
			filename = 'logistic_regression_model.sav'
			model = pickle.load(open(filename, 'rb'))
		except:
			model = self.result.best_estimator_

		minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
		minmax.fit(self.x).transform(self.x)

		x_test_std=minmax.fit_transform(self.x)
		y_pred=model.predict(x_test_std)



        
