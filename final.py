import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import time
import sys

class sales(object):
    """docstring for sales"""
    def __init__(self, trainFile, testFile):
        self.trainFile = trainFile
        self.testFile = testFile
        self.__xgb = XGBRegressor(max_depth=30, n_estimators=1500,min_child_weight=300,colsample_bytree=0.9,subsample=0.9,eta=0.15,seed=0)
        self.__rforest = RandomForestClassifier()
        self.train_data = None
        self.train_labels = None
        self.train_dat = None
        self.train_label = None
        self.test_data = None
        self.test_labels = None
        self.predicted_labels = None
        self.val_data = None
        self.val_labels = None
        self.x_val = None
        self.y_val = None

    
    def trainingData(self):
        df = pd.read_csv(self.trainFile)
        df = df.dropna()
        df['month']=pd.DatetimeIndex(df['date']).month
        df['day']=pd.DatetimeIndex(df['date']).day
        # df['year']=pd.DatetimeIndex(df['date']).year
        sub_set=['date','date_block_num','shop_id','item_id','item_price','item_cnt_day']
        df.drop_duplicates(sub_set, keep='first', inplace=True)
        df['item_cnt_day']=df['item_cnt_day'].clip(0,1100)
        y=df[['item_cnt_day']]
        self.train_label = df['item_cnt_day']
        df = df.drop(columns='item_cnt_day')
        df = df.drop(columns='ID')
        df = df.drop(columns='date')
        self.train_dat=df[['day','date_block_num','month','shop_id','item_id','item_price']]
        self.train_data,self.val_data,self.train_labels,self.val_labels=train_test_split(self.train_dat,self.train_label,test_size=0.2)
        self.x_val = self.train_data[-100:]
        self.y_val = self.train_labels[-100:]

    def testingData(self):
        df = pd.read_csv(self.testFile)
        df = df.dropna()
        df['month']=pd.DatetimeIndex(df['date']).month
        df['day']=pd.DatetimeIndex(df['date']).day
        # df['year']=pd.DatetimeIndex(df['date']).year
        sub_set=['date','date_block_num','shop_id','item_id','item_price']
        df.drop_duplicates(sub_set, keep='first', inplace=True)
        df=df.drop(['ID'],axis=1)
        df=df.drop(['date'],axis=1)
        df=df[['day','date_block_num','month','shop_id','item_id','item_price']]
        self.test_data = df

    def data(self):
        self.trainingData()
        self.testingData()

    def trainRandomForrest(self):
        self.__rforest.fit(self.train_data,self.train_labels)
    
    def testRandomForrest(self):
        self.test_labels = self.__rforest.predict(self.test_data)
        self.predicted_labels = self.__rforest.predict(self.val_data)
        print ("Random forest RMSE:  " + str(rmse(self.predicted_labels,self.val_labels)))

    def trainXGB(self):
        self.__xgb.fit(self.train_data,self.train_labels)

    def testXGB(self):
        self.test_labels = self.__xgb.predict(self.test_data)
        self.predicted_labels = self.__xgb.predict(self.val_data)
        print ("XGB RMSE:  " + str(rmse(self.predicted_labels,self.val_labels)))


if __name__ == "__main__":
    train_data_name = sys.argv[1]
    test_data_name = sys.argv[2]
    model = sales(train_data_name,test_data_name)
    model.data()
    # model.trainLogisticRegression()
    # model.testLogisticRegression()
    # plotConfusionMatrix(model.test_labels,model.predicted_labels)
    
    # model.trainDecesionTree()
    # model.testDecesionTree()

    model.trainRandomForrest()
    model.testRandomForrest()

    # model.trainXGB()
    # model.testXGB()



		




        

    

        