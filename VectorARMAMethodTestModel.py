# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""
import pandas as pd
from VectorARMAMethodUtils import (readVectorARMAMethodModel, 
                                   readVectorARMAMethodXTrain,
                                   saveVectorARMAMethodForecastedValues,
                                   importVectorARMAMethodDataset, 
                                   readVectorARMAMethodForecastedValues)

from VectorARMAMethodVisualization import (visualizeVectorARMAMethodPredictedValuesForMoney,
                                           visualizeVectorARMAMethodPredictedValuesForSpending)

"""
test the model on testing dataset
"""
def testVectorARMAMethodModel():
    
    #reading the full dataset
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    #reading model from pickle file
    vectorARMAMethodModel = readVectorARMAMethodModel()
    
    #Unlike the VARMAX model we'll use in upcoming sections, the VAR .forecast() function 
    #requires that we pass in a lag order number of previous observations as well. 
    #Unfortunately this forecast tool doesn't provide a DateTime index - we'll have to do that manually.
    #forecast for next 12 months
    vectorARMAMethodForecastedValues = vectorARMAMethodModel.forecast(12)
        
    numberOfObsevation = 12
    
    #Invert the transformation
    #Remember that the forecasted values represent second-order differences. 
    #To compare them to the original data we have to roll back each difference. 
    #To roll back a first-order difference we take the most recent value on the training side of the original series, 
    #and add it to a cumulative sum of forecasted values. 
    #When working with second-order differences we first must perform this operation on the most recent first-order difference.
    
    # Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
    vectorARMAMethodForecastedValues['Money1d'] = (vectorARMAMethodDataset['Money'].iloc[-numberOfObsevation-1]-vectorARMAMethodDataset['Money'].iloc[-numberOfObsevation-2]) + vectorARMAMethodForecastedValues['Money'].cumsum()
    
    # Now build the forecast values from the first difference set
    vectorARMAMethodForecastedValues['MoneyForecast'] = vectorARMAMethodDataset['Money'].iloc[-numberOfObsevation-1] + vectorARMAMethodForecastedValues['Money'].cumsum()
    
    # Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
    vectorARMAMethodForecastedValues['Spending1d'] = (vectorARMAMethodDataset['Spending'].iloc[-numberOfObsevation-1]-vectorARMAMethodDataset['Spending'].iloc[-numberOfObsevation-2]) + vectorARMAMethodForecastedValues['Spending'].cumsum()
    
    # Now build the forecast values from the first difference set
    vectorARMAMethodForecastedValues['SpendingForecast'] = vectorARMAMethodDataset['Spending'].iloc[-numberOfObsevation-1] + vectorARMAMethodForecastedValues['Spending'].cumsum()    
    #saving the foreasted values
    saveVectorARMAMethodForecastedValues(vectorARMAMethodForecastedValues)    

def plotVectorARMAMethodPredictedValuesForMoney():
    
    #reading the forecasted values
    vectorARMAMethodForecastedValues = readVectorARMAMethodForecastedValues()
    
    #reading the full dataset
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    visualizeVectorARMAMethodPredictedValuesForMoney(vectorARMAMethodDataset, vectorARMAMethodForecastedValues)

def plotVectorARMAMethodPredictedValuesForSpending():
    
    #reading the forecasted values
    vectorARMAMethodForecastedValues = readVectorARMAMethodForecastedValues()
    
    #reading the full dataset
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
        
    visualizeVectorARMAMethodPredictedValuesForSpending(vectorARMAMethodDataset, vectorARMAMethodForecastedValues)
    
if __name__ == "__main__":
    #testVectorARMAMethodModel()
    #plotVectorARMAMethodPredictedValuesForMoney()
    plotVectorARMAMethodPredictedValuesForSpending()