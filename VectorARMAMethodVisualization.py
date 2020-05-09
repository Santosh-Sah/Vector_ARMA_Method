# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab

def visualizeVectorARMAMethodPredictedValuesForMoney(vectorARMAMethodDataset, vectorARMAMethodForecastedValues):
    
    numberOfObservation = 12
    
    #plotting the predicted values
    vectorARMAMethodDataset['Money'][-numberOfObservation:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
    vectorARMAMethodForecastedValues['MoneyForecast'].plot(legend=True);
    
    pylab.savefig('PredeictedValuesForMoney.png')

def visualizeVectorARMAMethodPredictedValuesForSpending(vectorARMAMethodDataset, vectorARMAMethodForecastedValues):
    
    numberOfObservation = 12
    
    #plotting the predicted values
    vectorARMAMethodDataset['Spending'][-numberOfObservation:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
    vectorARMAMethodForecastedValues['SpendingForecast'].plot(legend=True);
    
    pylab.savefig('PredeictedValuesForSpending.png')

def visualizeVectorARMAMethodForecastedValues(vectorARMAMethodDataset, vectorARMAMethodForecastedValues):
    
    #plotting the forecated values with full dataset
    vectorARMAMethodDataset["PopEst"].plot()
    
    vectorARMAMethodForecastedValues.plot()
    
    pylab.savefig('ForecastedValues.png')

def visualizeSourceDataPlot(vectorARMAMethodDataset):
    
    #plotting the source dataset
    title = 'M2 Money Stock vs. Personal Consumption Expenditures'
    
    ylabel='Billions of dollars'
    
    xlabel='' 

    ax = vectorARMAMethodDataset['Spending'].plot(figsize=(16,5),title=title, legend = True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    vectorARMAMethodDataset['Money'].plot(legend=True)
    
    pylab.savefig('SourceDatasetPlot.png')

def visualizeResultPlots(vectorARMAMethodModel):
    
    vectorARMAMethodModel.plot()
    
    pylab.savefig('VARMAResultsPlot.png')

def visualizeForecastedPlots(vectorARMAMethodModel):
    
    vectorARMAMethodModel.plot_forecast(12)
    
    pylab.savefig('VARMAForecastedPlot.png')