# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:13:14 2021

@author: Abhinav.Bajpai
"""

import gc
##Clear variable/objects from workspace to free up memory
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
dir()
import numpy as np 
import pandas as pd
import matplotlib as mp
import seaborn as sns
print('The seaborn version is {}.'.format(sns.__version__))
import os
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import zscore
import datetime 
from scipy.stats import chi2_contingency ## for chi sq test 
from scipy.stats import   ttest_1samp,ttest_ind ##For t test
from scipy.stats import variation  
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from sklearn.preprocessing import StandardScaler ##For zscore scaling 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster ##Hierachical clustering
from sklearn.metrics import silhouette_samples, silhouette_score  ##kmeans sil score
from sklearn.tree import DecisionTreeClassifier   ##DT for CART
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
 ##for clustering
import time
from datetime import timedelta
from sklearn.cluster import KMeans  ## for k means
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.metrics import make_scorer
sns.set(color_codes=True) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings("ignore")
import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import re

import pickle
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from   statsmodels.tsa.api  import ExponentialSmoothing, SimpleExpSmoothing, Holt

from   sklearn.metrics                 import  mean_squared_error
import statsmodels.tools.eval_measures as      em
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The NUMPY version is {}.'.format(np.__version__))
print('The statsmodel version is {}.'.format(statsmodels.__version__))



emit_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 21-TimeSeries_MA_ETS\Emission.csv')
print(emit_orig.dtypes)

emit_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 21-TimeSeries_MA_ETS\Emission.csv',parse_dates=['Year-Month'])

##reading column as DATE  datatype
emit_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 21-TimeSeries_MA_ETS\Emission.csv',parse_dates=['Year-Month'],index_col='Year-Month')

##date = pd.date_range(start='1/1/1959', periods=len(df), freq='D')
emit=emit_orig.copy()
emit.drop(emit.columns[emit.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
emit=emit.dropna(how='all')
print(emit.shape)

emit.isnull().sum()

print(emit.head(5))
print(emit.dtypes)
print(emit.describe())
print(emit.info())

#plotting timeseries -Increase the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
emit.plot(grid=True)



##decompose additively 

emit_add_decomp=seasonal_decompose(emit,model="additive")
emit_add_decomp.plot()

emit_add_decomp.trend.plot()
emit_add_decomp.seasonal.plot()
emit_add_decomp.resid.plot()

##Decompose the time series multiplicatively
emit_mul_decompose = seasonal_decompose(emit, model = "multiplicative")
emit_mul_decompose.plot()

#Compare with the original series

 ##log transformation to make mult series into additive 
emit_log=emit.copy()
emit_log['CO2 Emission']=np.log(emit)
emit_log['CO2 Emission']
emit_log.plot()
plt.subplot(2,1,1)
plt.title('Original Time Series')
plt.plot(emit)

plt.subplot(2,1,2)
plt.title('Log Transformed Time Series')
plt.plot(emit_log)
plt.tight_layout()

##########################################

#Let's change the monthly series to quarterly. This would require aggregation.
emit_q = emit.resample('Q').mean()
emit_q.plot(grid=True)

#Let's change the monthly series to daily. 
emit_daily = emit.resample('D').ffill()
emit_daily.plot(grid=True)

##train test split

trainset=emit[0:int(len(emit)*0.8)]
testset=emit[int(len(emit)*0.8):]
print(trainset.shape)
print(testset.shape)


##visualize training and test data 
trainset['CO2 Emission'].plot(fontsize=14)
testset['CO2 Emission'].plot(fontsize=14)
plt.grid()
plt.legend(['Training Data','Test Data'])
plt.show()


##Naive model 𝑦̂ 𝑡+1=𝑦𝑡 
NaiveModel_train = trainset.copy()
NaiveModel_test = testset.copy()
NaiveModel_test['naive'] = np.asarray(trainset['CO2 Emission'])[len(np.asarray(trainset['CO2 Emission']))-1]
NaiveModel_test['naive'].head()
##plot naiive model
plt.plot(NaiveModel_train['CO2 Emission'], label='Train')
plt.plot(testset['CO2 Emission'], label='Test')
plt.plot(NaiveModel_test['naive'], label='Naive Forecast on Test Data')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.grid();

##metric -RMSE 
rmse_model2_test = metrics.mean_squared_error(testset['CO2 Emission'],NaiveModel_test['naive'],squared=False)
print("For Naive MODEL,  RMSE is %3.3f" %(rmse_model2_test))

## Mean Absolute Percentage Error (MAPE) - Function Definition
def MAPE(y_true, y_pred):
    return np.mean((np.abs(y_true-y_pred))/(y_true))*100

mape_naive=MAPE(testset['CO2 Emission'],NaiveModel_test['naive'])
print(mape_naive)



##Moving average 
##For the moving average model, we are going to calculate rolling means 
##(or moving averages) for different intervals. 
##The best interval can be determined by the maximum accuracy (or the minimum error) over here.

MovingAverage=emit.copy()
MovingAverage['Trailing_2'] = MovingAverage['CO2 Emission'].rolling(2).mean()
MovingAverage['Trailing_4'] = MovingAverage['CO2 Emission'].rolling(4).mean()
MovingAverage['Trailing_6'] = MovingAverage['CO2 Emission'].rolling(6).mean()
MovingAverage['Trailing_9'] = MovingAverage['CO2 Emission'].rolling(9).mean()

## Plotting on the whole data
plt.plot(MovingAverage['CO2 Emission'], label='Main')
plt.plot(MovingAverage['Trailing_2'], label='2 Point Moving Average')
plt.plot(MovingAverage['Trailing_4'], label='4 Point Moving Average')
plt.plot(MovingAverage['Trailing_6'],label = '6 Point Moving Average')
plt.plot(MovingAverage['Trailing_9'],label = '9 Point Moving Average')
plt.legend(loc = 'best')
plt.grid();


#Creating train and test set 
trailing_MovingAverage_train=MovingAverage[0:int(len(MovingAverage)*0.7)] 
trailing_MovingAverage_test=MovingAverage[int(len(MovingAverage)*0.7):]


## Test Data - RMSE  --> 2 point Trailing MA
rmse_model4_test_2 = metrics.mean_squared_error(testset['CO2 Emission'],trailing_MovingAverage_test['Trailing_2'],squared=False)
print("For 2 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_2))

## Test Data - RMSE --> 4 point Trailing MA
rmse_model4_test_4 = metrics.mean_squared_error(testset['CO2 Emission'],trailing_MovingAverage_test['Trailing_4'],squared=False)
print("For 4 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_4))

## Test Data - RMSE --> 6 point Trailing MA

rmse_model4_test_6 = metrics.mean_squared_error(testset['CO2 Emission'],trailing_MovingAverage_test['Trailing_6'],squared=False)
print("For 6 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_6))

## Test Data - RMSE --> 9 point Trailing MA

rmse_model4_test_9 = metrics.mean_squared_error(testset['CO2 Emission'],trailing_MovingAverage_test['Trailing_9'],squared=False)
print("For 9 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_9))

resultsDf_4 = pd.DataFrame({'Test RMSE': [rmse_model4_test_2,rmse_model4_test_4
                                          ,rmse_model4_test_6,rmse_model4_test_9]}
                           ,index=['2pointTrailingMovingAverage','4pointTrailingMovingAverage'
                                   ,'6pointTrailingMovingAverage','9pointTrailingMovingAverage'])

resultsDf_4


##SES
model_SES = SimpleExpSmoothing(trainset,initialization_method='estimated')
# Fitting the Simple Exponential Smoothing model and asking python to choose the optimal parameters
model_SES_autofit = model_SES.fit(optimized=True)
print(model_SES_autofit.params)
# Using the fitted model on the training set to forecast on the test set
SES_predict = model_SES_autofit.forecast(steps=len(testset))
SES_predict
## Plotting the Training data, Test data and the forecasted values
plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')
plt.plot(SES_predict, label='Simple Exponential Smoothing predictions on Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Simple Exp Smoothing');
##
print('SES RMSE:',mean_squared_error(testset.values,SES_predict.values,squared=False))
#different way to calculate RMSE
print('SES RMSE (calculated using statsmodels):',em.rmse(testset.values,SES_predict.values)[0])



##SES2 -on log  
emit_train_log=np.log(trainset)
emit_test_log=np.log(testset)
# emit_log['CO2 Emission']=np.log(emit)
# emit_log['CO2 Emission']
##SES2
model_SES = SimpleExpSmoothing(emit_train_log,initialization_method='estimated')
# Fitting the Simple Exponential Smoothing model and asking python to choose the optimal parameters
model_SES_autofit = model_SES.fit(optimized=True)
print(model_SES_autofit.params)
# Using the fitted model on the training set to forecast on the test set
SES_predict2 = model_SES_autofit.forecast(steps=len(emit_test_log))
SES_predict2
## Plotting the Training data, Test data and the forecasted values
plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')
plt.plot(SES_predict, label='Simple Exponential Smoothing predictions on Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Simple Exp Smoothing');
##
print('SES RMSE:',mean_squared_error(emit_test_log.values,SES_predict2.values,squared=False))
#different way to calculate RMSE
print('SES RMSE (calculated using statsmodels):',em.rmse(emit_test_log.values,SES_predict2.values)[0])



##DES-Holt linear method -ETS(A, A, N) - Holt's linear method with additive errors

# Initializing the Double Exponential Smoothing Model
model_DES = Holt(trainset,initialization_method='estimated')
# Fitting the model
model_DES = model_DES.fit(optimized=True)
print('')
print('==Holt model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_DES.params)

# Forecasting using this model for the duration of the test set
DES_predict =  model_DES.forecast(len(testset))
DES_predict

## Plotting the Training data, Test data and the forecasted values
plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')
plt.plot(SES_predict, label='SES')
plt.plot(DES_predict, label='DES')
plt.legend(loc='best')
plt.grid()
plt.title('Simple and Double Exponential Smoothing Predictions');

print('DES RMSE:',mean_squared_error(testset.values,DES_predict.values,squared=False))


##HoltWinter ETS(A,A,A)
# Initializing the Double Exponential Smoothing Model
model_TES = ExponentialSmoothing(trainset,trend='additive',seasonal='additive',initialization_method='estimated')
# Fitting the model
model_TES = model_TES.fit()

print('')
print('==Holt Winters model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_TES.params)

# Forecasting using this model for the duration of the test set
TES_predict =  model_TES.forecast(len(testset))
TES_predict

## Plotting the Training data, Test data and the forecasted values

plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')

plt.plot(SES_predict, label='SES')
plt.plot(DES_predict, label='DES')
plt.plot(TES_predict, label='HW -Additive')
plt.legend(loc='best')
plt.grid()
plt.title('Simple,Double and Triple Exponential Smoothing Predictions');


print('TES RMSE:',mean_squared_error(testset.values,TES_predict.values,squared=False))

##Holt-Winters - ETS(A, A, M) - Holt Winter's linear method
# Initializing the Double Exponential Smoothing Model
model_TES_am = ExponentialSmoothing(trainset,trend='add',seasonal='multiplicative',initialization_method='estimated')
# Fitting the model
model_TES_am = model_TES_am.fit()

print('')
print('==Holt Winters model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_TES_am.params)

# Forecasting using this model for the duration of the test set
TES_predict_am =  model_TES_am.forecast(len(testset))
TES_predict_am

## Plotting the Training data, Test data and the forecasted values

plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')
plt.plot(SES_predict, label='SES')
plt.plot(DES_predict, label='DES')
plt.plot(TES_predict, label='HW Add')
plt.plot(TES_predict_am, label='HW Mul')
plt.legend(loc='best')
plt.grid()
plt.title('Simple,Double and Triple Exponential Smoothing Predictions');


print('TES_am RMSE:',mean_squared_error(testset.values,TES_predict_am.values,squared=False))





