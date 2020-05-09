# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from VectorARMAMethodUtils import (importVectorARMAMethodDataset, saveTrainingAndTestingDataset, 
                                   splitVectorARMAMethodDataset, agumentedDickeyFullerTest)

def preprocess():
    
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    #taking the first difference
    vectorARMAMethodDatasetFirstDiff = vectorARMAMethodDataset.diff()
    
    #taking the second difference. Second difference data is stationary
    vectorARMAMethodDatasetSecondDiff = vectorARMAMethodDatasetFirstDiff.diff()
    
    #dropping missing values
    vectorARMAMethodDatasetSecondDiff = vectorARMAMethodDatasetSecondDiff.dropna()
    
    X_train, X_test = splitVectorARMAMethodDataset(vectorARMAMethodDatasetSecondDiff)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    
def testIsDatasetStationary():
    
    vectorARMAMethodDataset = importVectorARMAMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
        
    #agumentedDickeyFullerTest(vectorAutoRegressiveMethodDataset["Money"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic        4.239022
#     p-value                   1.000000
#     # lags used               4.000000
#     # observations          247.000000
#     critical value (1%)      -3.457105
#     critical value (5%)      -2.873314
#     critical value (10%)     -2.573044
#     Weak evidence against the null hypothesis
#     Fail to reject the null hypothesis
#     Data has a unit root and is non-stationary
# =============================================================================
    
    #agumentedDickeyFullerTest(vectorAutoRegressiveMethodDataset["Spending"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic        0.149796
#     p-value                   0.969301
#     # lags used               3.000000
#     # observations          248.000000
#     critical value (1%)      -3.456996
#     critical value (5%)      -2.873266
#     critical value (10%)     -2.573019
#     Weak evidence against the null hypothesis
#     Fail to reject the null hypothesis
#     Data has a unit root and is non-stationary
# =============================================================================
    
    vectorARMAMethodDatasetFirstDiff = vectorARMAMethodDataset.diff()
    
    #agumentedDickeyFullerTest(vectorAutoRegressiveMethodDatasetFirstDiff["Money"], title = "Money First Differnce")
    
# =============================================================================
#     Augmented Dickey-Fuller Test: Money First Differnce
#     ADF test statistic       -2.057404
#     p-value                   0.261984
#     # lags used              15.000000
#     # observations          235.000000
#     critical value (1%)      -3.458487
#     critical value (5%)      -2.873919
#     critical value (10%)     -2.573367
#     Weak evidence against the null hypothesis
#     Fail to reject the null hypothesis
#     Data has a unit root and is non-stationary
# =============================================================================
    
    #agumentedDickeyFullerTest(vectorAutoRegressiveMethodDatasetFirstDiff["Spending"], title = "Spending First Differnce")
    
# =============================================================================
#     Augmented Dickey-Fuller Test: Spending First Differnce
#     ADF test statistic     -7.226974e+00
#     p-value                 2.041027e-10
#     # lags used             2.000000e+00
#     # observations          2.480000e+02
#     critical value (1%)    -3.456996e+00
#     critical value (5%)    -2.873266e+00
#     critical value (10%)   -2.573019e+00
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
    vectorARMAMethodDatasetSecondDiff = vectorARMAMethodDatasetFirstDiff.diff()
    
    agumentedDickeyFullerTest(vectorARMAMethodDatasetSecondDiff["Money"], title = "Money Second Differnce")
    
# =============================================================================
#     Augmented Dickey-Fuller Test: Money Second Differnce
#     ADF test statistic     -7.077471e+00
#     p-value                 4.760675e-10
#     # lags used             1.400000e+01
#     # observations          2.350000e+02
#     critical value (1%)    -3.458487e+00
#     critical value (5%)    -2.873919e+00
#     critical value (10%)   -2.573367e+00
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# ============================================================================
    
    agumentedDickeyFullerTest(vectorARMAMethodDatasetSecondDiff["Spending"], title = "Spending Second Differnce")
    
# =============================================================================
#     Augmented Dickey-Fuller Test: Spending Second Differnce
#     ADF test statistic     -8.760145e+00
#     p-value                 2.687900e-14
#     # lags used             8.000000e+00
#     # observations          2.410000e+02
#     critical value (1%)    -3.457779e+00
#     critical value (5%)    -2.873609e+00
#     critical value (10%)   -2.573202e+00
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
    
if __name__ == "__main__":
    testIsDatasetStationary()
    #preprocess()