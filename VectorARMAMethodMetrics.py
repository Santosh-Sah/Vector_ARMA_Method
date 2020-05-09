# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from statsmodels.tools.eval_measures import rmse

from VectorARMAMethodUtils import (importVectorARMAMethodDataset, readVectorARMAMethodForecastedValues)

"""

calculating VectorARMAMethod metrics

"""
def testVectorARMAMethodMetrics():
    
    numberOfObservation = 12
    
    #reading the full dataset
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")

    #reading the forecasted values
    vectorARMAMethodForecastedValues = readVectorARMAMethodForecastedValues()    
    
    #rmse for money
    rmseForMoneyForecasting = rmse(vectorARMAMethodDataset["Money"][-numberOfObservation:], vectorARMAMethodForecastedValues["MoneyForecast"])
    
    #rmse for spending
    rmseForSpendingForecasting = rmse(vectorARMAMethodDataset["Spending"][-numberOfObservation:], vectorARMAMethodForecastedValues["SpendingForecast"])
    
    print(rmseForMoneyForecasting) #423.65406365399417
    
    print(rmseForSpendingForecasting) #243.5870145647394
    
    
if __name__ == "__main__":
    testVectorARMAMethodMetrics()