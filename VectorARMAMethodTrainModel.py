# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.statespace.varmax import VARMAX
from pmdarima import auto_arima

from VectorARMAMethodUtils import (saveVectorARMAMethodModel, readVectorARMAMethodXTrain, 
                                   importVectorARMAMethodDataset, saveVectorARMAMethodModelForFullDataset)

from VectorARMAMethodVisualization import (visualizeSourceDataPlot)

"""
Train VectorARMAMethod model on training set
"""
def trainVectorARMAMethodModel():
    
    X_train = readVectorARMAMethodXTrain()
    
    #training model on the training set
    vectorARMAMethodModel = VARMAX(X_train, order = (1,2), trend = "c") 
    
    #we are taking p = 5 as we have created different models based on the different p values.
    #Model gives minimum aic and bic for p =5
    vectorARMAMethodModelResult = vectorARMAMethodModel.fit(maxiter=1000, disp=False)
    
    #saving the model in pickle file
    saveVectorARMAMethodModel(vectorARMAMethodModelResult)
    
    print(vectorARMAMethodModelResult.summary())
    
# =============================================================================
#                              Statespace Model Results
#     =================================================================================
#     Dep. Variable:     ['Money', 'Spending']   No. Observations:                  238
#     Model:                        VARMA(1,2)   Log Likelihood               -2286.151
#                                  + intercept   AIC                           4606.303
#     Date:                   Sat, 09 May 2020   BIC                           4665.331
#     Time:                           22:23:52   HQIC                          4630.092
#     Sample:                       03-01-1995
#                                 - 12-01-2014
#     Covariance Type:                     opg
#     ===================================================================================
#     Ljung-Box (Q):                67.82, 27.95   Jarque-Bera (JB):       566.08, 127.11
#     Prob(Q):                        0.00, 0.92   Prob(JB):                   0.00, 0.00
#     Heteroskedasticity (H):         5.69, 2.90   Skew:                      1.35, -0.35
#     Prob(H) (two-sided):            0.00, 0.00   Kurtosis:                  10.06, 6.51
#                                 Results for equation Money
#     ==================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
#     ----------------------------------------------------------------------------------
#     intercept          0.2505      0.908      0.276      0.783      -1.528       2.030
#     L1.Money          -1.2265      4.200     -0.292      0.770      -9.458       7.005
#     L1.Spending        2.0165      6.556      0.308      0.758     -10.833      14.866
#     L1.e(Money)        0.4562      4.191      0.109      0.913      -7.759       8.671
#     L1.e(Spending)    -2.1381      6.560     -0.326      0.744     -14.995      10.718
#     L2.e(Money)       -1.4543      4.184     -0.348      0.728      -9.655       6.746
#     L2.e(Spending)     1.8683      5.756      0.325      0.746      -9.414      13.151
#                               Results for equation Spending
#     ==================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
#     ----------------------------------------------------------------------------------
#     intercept          0.0915      0.200      0.457      0.647      -0.301       0.484
#     L1.Money          -0.4414      2.836     -0.156      0.876      -6.000       5.117
#     L1.Spending        0.8097      4.239      0.191      0.849      -7.499       9.118
#     L1.e(Money)        0.5685      2.892      0.197      0.844      -5.099       6.236
#     L1.e(Spending)    -1.7718      4.218     -0.420      0.674     -10.038       6.495
#     L2.e(Money)       -0.5690      2.893     -0.197      0.844      -6.239       5.101
#     L2.e(Spending)     0.7877      3.679      0.214      0.830      -6.423       7.999
#                                       Error covariance matrix
#     ===========================================================================================
#                                   coef    std err          z      P>|z|      [0.025      0.975]
#     -------------------------------------------------------------------------------------------
#     sqrt.var.Money             25.6665      2.930      8.761      0.000      19.925      31.408
#     sqrt.cov.Money.Spending   -10.0296      2.200     -4.559      0.000     -14.342      -5.718
#     sqrt.var.Spending          33.3827      1.286     25.963      0.000      30.863      35.903
#     ===========================================================================================
# =============================================================================

def plotTheSourceData():
    
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    visualizeSourceDataPlot(vectorARMAMethodDataset)

def determineOrderOfPQForMoney():
    
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    vectorARMAMethodDataset = vectorARMAMethodDataset.dropna()
    
    autoArima = auto_arima(vectorARMAMethodDataset["Money"], maxiter = 1000)
    
    print(autoArima.summary())

def determineOrderOfPQForSpending():
    
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    vectorARMAMethodDataset = vectorARMAMethodDataset.dropna()
    
    autoArima = auto_arima(vectorARMAMethodDataset["Spending"], maxiter = 1000)
    
    print(autoArima.summary())
    
    
        

if __name__ == "__main__":
    #plotTheSourceData()
    #determineOrderOfPQForMoney()
    #determineOrderOfPQForSpending()
    trainVectorARMAMethodModel() 
