# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importVectorARMAMethodDataset(vectorARMAMethodDatasetFileName1, vectorARMAMethodDatasetFileName2):
    
    vectorARMAMethodDataset1 = pd.read_csv(vectorARMAMethodDatasetFileName1,index_col=0, parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    vectorARMAMethodDataset1.index.freq = "MS"
    
    vectorARMAMethodDataset2 = pd.read_csv(vectorARMAMethodDatasetFileName2,index_col=0, parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    vectorARMAMethodDataset2.index.freq = "MS"
    
    vectorARMAMethodDatasetFinalDataset = vectorARMAMethodDataset1.join(vectorARMAMethodDataset2)
    
    return vectorARMAMethodDatasetFinalDataset

#splitting dataset into training and testing set
def splitVectorARMAMethodDataset(vectorARMAMethodDataset):
    
    #splitting the dataset into training and testing set.
    vectorARMAMethodTrainingSet = vectorARMAMethodDataset.iloc[0:-12]
    vectorARMAMethodTestingSet = vectorARMAMethodDataset.iloc[-12:]
    
    return vectorARMAMethodTrainingSet, vectorARMAMethodTestingSet

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readVectorARMAMethodXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readVectorARMAMethodXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save VectorARMAMethod as a pickle file.
"""
def saveVectorARMAMethodModel(vectorARMAMethodModel):
    
    #Write VectorARMAMethodModel as a picke file
    with open("VectorARMAMethodModel.pkl",'wb') as vectorARMAMethodModel_Pickle:
        pickle.dump(vectorARMAMethodModel, vectorARMAMethodModel_Pickle, protocol = 2)

"""
read VectorARMAMethod from pickle file
"""
def readVectorARMAMethodModel():
    
    #load VectorARMAMethodModel model
    with open("VectorARMAMethodModel.pkl","rb") as vectorARMAMethodModel:
        vectorARMAMethodModel = pickle.load(vectorARMAMethodModel)
    
    return vectorARMAMethodModel

"""
Save VectorARMAMethod as a pickle file.
"""
def saveVectorARMAMethodModelForFullDataset(vectorARMAMethodModelForFullDataset):
    
    #Write VectorARMAMethodModelForFullDataset as a picke file
    with open("VectorARMAMethodModelForFullDataset.pkl",'wb') as vectorARMAMethodModelForFullDataset_Pickle:
        pickle.dump(vectorARMAMethodModelForFullDataset, vectorARMAMethodModelForFullDataset_Pickle, protocol = 2)

"""
read VectorARMAMethod from pickle file
"""
def readVectorARMAMethodModelForFullDataset():
    
    #load VectorARMAMethodModelForFullDataset model
    with open("VectorARMAMethodModelForFullDataset.pkl","rb") as vectorARMAMethodModelForFullDataset:
        vectorARMAMethodModelForFullDataset = pickle.load(vectorARMAMethodModelForFullDataset)
    
    return vectorARMAMethodModelForFullDataset

"""
save VectorARMAMethodPredictedValues as a pickle file
"""

def saveVectorARMAMethodPredictedValues(vectorARMAMethodPredictedValues):
    
    #Write VectorARMAMethodPredictedValues in a picke file
    with open("VectorARMAMethodPredictedValues.pkl",'wb') as vectorARMAMethodPredictedValues_Pickle:
        pickle.dump(vectorARMAMethodPredictedValues, vectorARMAMethodPredictedValues_Pickle, protocol = 2)

"""
read VectorARMAMethodPredictedValues from pickle file
"""
def readVectorARMAMethodPredictedValues():
    
    #load VectorARMAMethodPredictedValues
    with open("VectorARMAMethodPredictedValues.pkl","rb") as vectorARMAMethodPredictedValues_pickle:
        vectorARMAMethodPredictedValues = pickle.load(vectorARMAMethodPredictedValues_pickle)
    
    return vectorARMAMethodPredictedValues

"""
save VectorARMAMethodForecastedValues as a pickle file
"""

def saveVectorARMAMethodForecastedValues(vectorARMAMethodForecastedValues):
    
    #Write VectorARMAMethodForecastedValues in a picke file
    with open("VectorARMAMethodForecastedValues.pkl",'wb') as vectorARMAMethodForecastedValues_Pickle:
        pickle.dump(vectorARMAMethodForecastedValues, vectorARMAMethodForecastedValues_Pickle, protocol = 2)

"""
read VectorARMAMethodForecastedValues from pickle file
"""
def readVectorARMAMethodForecastedValues():
    
    #load VectorARMAMethodForecastedValues
    with open("VectorARMAMethodForecastedValues.pkl","rb") as vectorARMAMethodForecastedValues_pickle:
        vectorARMAMethodForecastedValues = pickle.load(vectorARMAMethodForecastedValues_pickle)
    
    return vectorARMAMethodForecastedValues

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


