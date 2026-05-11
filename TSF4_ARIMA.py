# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:36:48 2021

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

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The NUMPY version is {}.'.format(np.__version__))
print('The statsmodel version is {}.'.format(statsmodels.__version__))


## Mean Absolute Percentage Error (MAPE) - Function Definition

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean((np.abs(y_true-y_pred))/(y_true))*100

## Importing the mean_squared_error function from sklearn to calculate the RMSE

from sklearn.metrics import mean_squared_error



sales_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 22-TimeSeries_ARIMA\Sales_quantity.csv')
print(sales_orig.dtypes)

sales=sales_orig.copy()
##creating a timestamp (should know first date )
time_stamp=pd.date_range(start='2015-01-01',periods=len(sales),freq='M')
##date = pd.date_range(start='1/1/1959', periods=len(df), freq='D')

time_stamp


sales['Time_Stamp']=time_stamp


sales.set_index(keys='Time_Stamp',inplace=True)

sales.drop(sales.columns[sales.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values

print(sales.shape)
sales.isnull().sum()
print(sales.head(5))
print(sales.dtypes)
print(sales.describe())
print(sales.info())

#plotting timeseries -Increase the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
sales.plot(grid=True)


##Decompose the time series multiplicatively
sales_mul_decompose = seasonal_decompose(sales, model = "multiplicative")
sales_mul_decompose.plot()


trend = sales_mul_decompose.trend
seasonality = sales_mul_decompose.seasonal
residual = sales_mul_decompose.resid

print('Trend','\n',trend.head(12),'\n')
print('Seasonality','\n',seasonality.head(12),'\n')
print('Residual','\n',residual.head(12),'\n')



# Check for stationarity of the whole Time Series data.
# The Augmented Dickey-Fuller test is an unit root test which determines 
#whether there is a unit root and subsequently whether the series is non-stationary.

# The hypothesis in a simple form for the ADF test is:

# 𝐻0  : The Time Series has a unit root and is thus non-stationary.
# 𝐻1  : The Time Series does not have a unit root and is thus stationary.
# We would want the series to be stationary for building ARIMA models and thus we would want the p-value of this test to be less than the  𝛼  value.

dftest = adfuller(sales,regression='ct')
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is' ,dftest[1])
print('Number of lags used' ,dftest[2])


##p-val ! < 0.05 ,hence fail to reject NULL hyp ,ie series is non stationary 
#We see that at 5% significant level the Time Series is non-stationary.


##Let us take one level of differencing to see whether the series becomes stationary.

dftest = adfuller(sales.diff().dropna(),regression='ct')
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is' ,dftest[1])
print('Number of lags used' ,dftest[2])


##plotting the differences 
sales.diff().dropna().plot(grid=True)

##plot ACF and PACF on whole data

plot_acf(sales,alpha=0.05);


##inferences ???
plot_pacf(sales,zero=False,alpha=0.05);

plot_pacf(sales,zero=False,alpha=0.05,method='ywmle');

##inference ??

##train test split 
sales.index.year.unique()

trainset = sales[sales.index<='2019'] 
testset = sales[sales.index>'2019']

print(trainset.shape)
print(testset.shape)


trainset.plot(grid=True);



dftest = adfuller(trainset,regression='ct')
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is' ,dftest[1])
print('Number of lags used' ,dftest[2])

trainset.info()


##ARIMA -grid search 

## The following loop helps us in getting a combination of different parameters of p and q in the range of 0 and 2
## We have kept the value of d as 1 as we need to take a difference of the series to make it stationary.

import itertools
p = q = range(0, 4)
d= range(1,2)
pdq = list(itertools.product(p, d, q))
print('Examples of the parameter combinations for the Model')
for i in range(0,len(pdq)):
    print('Model: {}'.format(pdq[i]))

# Creating an empty Dataframe with column names only
ARIMA_AIC = pd.DataFrame(columns=['param', 'AIC'])
ARIMA_AIC

for param in pdq:# running a loop within the pdq parameters defined by itertools
    ARIMA_model = ARIMA(trainset['Sales_quantity'].values,order=param).fit()#fitting the ARIMA model
    #using the parameters from the loop
    print('ARIMA{} - AIC:{}'.format(param,ARIMA_model.aic))#printing the parameters and the AIC
    #from the fitted models
    ARIMA_AIC = ARIMA_AIC.append({'param':param, 'AIC': ARIMA_model.aic}, ignore_index=True)
    #appending the AIC values and the model parameters to the previously created data frame
    #for easier understanding and sorting of the AIC values
    
    
## Sort the above AIC values in the ascending order to get the 
##parameters for the minimum AIC value

ARIMA_AIC.sort_values(by='AIC',ascending=True).head()

##creating on best params
auto_ARIMA = ARIMA(trainset, order=(2,1,2))

results_auto_ARIMA = auto_ARIMA.fit()

print(results_auto_ARIMA.summary())

##diagnostics plot
results_auto_ARIMA.plot_diagnostics();


##prediction 

predicted_auto_ARIMA = results_auto_ARIMA.forecast(steps=len(testset))


rmse = mean_squared_error(testset['Sales_quantity'],predicted_auto_ARIMA,squared=False)
##mape_1 = mean_absolute_percentage_error(testset['Sales_quantity'],predicted_auto_ARIMA)
##print('RMSE:',rmse,'\nMAPE:',mape)

##SARIMA -grid search 
plot_acf(trainset.diff(),title='Training Data Autocorrelation',missing='drop');


import itertools
p = q = range(0, 4)
d= range(1,2)
D = range(0,1)
pdq = list(itertools.product(p, d, q))
PDQ = [(x[0], x[1], x[2], 6) for x in list(itertools.product(p, D, q))]
print('Examples of the parameter combinations for the Model are')
for i in range(1,len(pdq)):
    print('Model: {}{}'.format(pdq[i], PDQ[i]))



SARIMA_AIC = pd.DataFrame(columns=['param','seasonal', 'AIC'])
SARIMA_AIC


import statsmodels.api as sm

for param in pdq:
    for param_seasonal in PDQ:
        SARIMA_model = sm.tsa.statespace.SARIMAX(trainset['Sales_quantity'].values,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

        results_SARIMA = SARIMA_model.fit(maxiter=1000)
        print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
        SARIMA_AIC = SARIMA_AIC.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMA.aic}, ignore_index=True)




SARIMA_AIC.sort_values(by=['AIC']).head()

import statsmodels.api as sm

auto_SARIMA = sm.tsa.statespace.SARIMAX(trainset['Sales_quantity'],
                                order=(1, 1, 3),
                                seasonal_order=(3, 0, 3, 6),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_auto_SARIMA = auto_SARIMA.fit(maxiter=1000)
print(results_auto_SARIMA.summary())


results_auto_SARIMA.plot_diagnostics();


##predict SARIMA 

predicted_auto_SARIMA = results_auto_SARIMA.get_forecast(steps=len(testset))

predicted_auto_SARIMA.summary_frame(alpha=0.05).head()

rmse = mean_squared_error(testset['Sales_quantity'],predicted_auto_SARIMA.predicted_mean,squared=False)
##mape = mean_absolute_percentage_error(test['Sales_quantity'],predicted_auto_SARIMA.predicted_mean)
##print('RMSE:',rmse,'\nMAPE:',mape)

##temp_resultsDf = pd.DataFrame({'RMSE': rmse,'MAPE':mape}
                    ##       ,index=['SARIMA(1,1,3)(3,0,3,6)'])



