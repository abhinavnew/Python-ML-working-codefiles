# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:50:18 2021

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


def getupperlower_outlier(col):
    sorted(col)
    if col.isnull().values.any()==True:
        Q1,Q3,Q1_0,Q99,Q5,Q95=np.nanpercentile(col,[25,75,1,99,5,95])
        count=col.isnull().sum()
        print("There are NaN values in the column,which we will ignore-No of NaNs=",count)
    else:
        Q1,Q3,Q1_0,Q99,Q5,Q95=np.percentile(col,[25,75,1,99,5,95])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    print("Inside function:Lower range = ",lower_range)
    print("Inside function:Upper range = ",upper_range)
    print("Inside function:1st percentile = ",Q1_0)
    print("Inside function:99th percentile = ",Q99)
    print("Inside function:5th percentile = ",Q5)
    print("Inside function:95th percentile = ",Q95)
    return lower_range, upper_range



def treat_outlier_5_95_winsor(x):
    # taking 5,25,75 percentile of column
    q5= np.percentile(x,5)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    dt=np.percentile(x,95)
    #calculationg IQR range
    IQR=q75-q25
    print("INSIDE FUNCTION :Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: dt if y > upper_bound else y).apply(lambda y: q5 if y < lower_bound else y)

def treat_outlier_1_99_winsor(x):
    # taking 1,25,75,99 percentile of column
    q1=np.percentile(x,1)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    q99=np.percentile(x,99)
    #calculationg IQR range
    IQR=q75-q25
    print("INSIDE FUNCTION :Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: q99 if y > upper_bound else y).apply(lambda y: q1 if y < lower_bound else y)


def treat_outlier_ul_ll_winsor(x):
    # taking 5,25,75 percentile of column
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
   #calculationg IQR range
    IQR=q75-q25
    print("INSIDE FUNCTION:Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: upper_bound if y > upper_bound else y).apply(lambda y: lower_bound if y < lower_bound else y)



def missvalue_replacenull(dframe,cat_colnames):
    for i in cat_colnames :
        a=dframe[dframe[i].str.contains('\?') | dframe[i].str.contains("NA") | dframe[i].str.contains("NaN")]
        print("records containing ? NA NAN etc are  ",a.shape )
        print("For column name =",i)
           
    
def get_num_cat_colnames(dframe):
        hl_cat=[]
        hl_num=[]
        for i in dframe.columns :
            if dframe[i].dtype=="object":
                hl_cat.append(i)
            else :
                hl_num.append(i)
        print("Inside function :Categorical cols are =",hl_cat)
        print("Inside function :Numerical cols are =",hl_num)
        return hl_cat,hl_num
    
    
 ##zero variance or constant features
def return_constant_columns(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    novar_columns = dataframe.columns[dataframe.nunique()<=1]
    print("no Variance columns are ",novar_columns)
    a=dataframe.loc[:,novar_columns]
    print(a.shape)
    return a


def select_badvalues_rows(dframe,cat_colnames,bad_vals):
    smalldfs=[]
    for i in cat_colnames :           
        smalldfs.append(dframe.loc[dframe[i].str.contains('|'.join(bad_values))==True])
        print("For column name =",i)
    print(type(smalldfs))
    largedf=pd.concat(smalldfs,ignore_index=True)    
    return largedf


## Test for stationarity of the series - Dicky Fuller test

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean() #determining the rolling mean
    rolstd = timeseries.rolling(window=7).std()   #determining the rolling standard deviation

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput,'\n')

# c=select_badvalues_rows(df1,cat_col,bad_values)
# cat_col,num_col=get_num_cat_colnames(df1)
    

##Simple Imputer 
# #from sklearn.preprocessing import Imputer
# #my_imputer = Imputer()
# #data_with_imputed_values = my_imputer.fit_transform(original_data)

# from sklearn.impute import SimpleImputer
# rep_0 = SimpleImputer(missing_values=0, strategy="mean")
# cols=x_train.columns
# x_train = pd.DataFrame(rep_0.fit_transform(x_train))
# x_test = pd.DataFrame(rep_0.fit_transform(x_test))

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 30)

pd.set_option("display.max_colwidth",100)
##Imputing badvalues 
# df['col1'].replace('?',np.NaN,inplace=True)
# df['col1']=df['col1'].astype('float64')
# ##check and replace with col3 if condition true else let it be col2
# df['col2']=np.where(df['col2']==-1,df['col3'],df['col2'])
# ##impute with mean when bad val -1
# df['col5'].replace(-1,df['col5'].mean(),inplace=True)
# ##impute with mean/median for missing values (if no outliers then safe)
# df.col6.fillna(df.col6.mean(),inplace=True)
# from sklearn.impute import SimpleImputer 
# imputer=SimpleImputer(missing_values=np.nan,strategy='median')
# df_numericcols=pd.DataFrame(imputer.fit_transform(df_numericcols),columns=df_numericcols.columns)

##Basic EDA ::Columns dtypes,shape,unique values 
##Basic EDA ::Which Features are Categorical (Character)Or Categorical (Numerical),Which are continuous
##correlated columns,same value or Zero variance columns;; DOM -SMET (Duplicates,outliers,missing-Scaling mlticollinr,encoding,transformation)

####File Read and basic EDA 
rose_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\TimeSeries_Project\Rose.csv')
print(rose_orig.dtypes)
print(rose_orig.shape)
print(rose_orig.head())

rose=rose_orig.copy()
##creating a timestamp (should know first date )
time_stamp=pd.date_range(start='1980-01-01',periods=len(rose),freq='M')
##date = pd.date_range(start='1/1/1959', periods=len(df), freq='D')

print(time_stamp)
rose['Time_Stamp']=time_stamp
rose.set_index(keys='Time_Stamp',inplace=True)

rose.head()

rose.drop(rose.columns[rose.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
rose.shape
##Droping all rows with NA/NULL values
rose.drop(["YearMonth"],axis=1,inplace=True)

print(rose.info())
print(rose.columns)
print(rose.describe())

#plotting timeseries -Increase the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rose.plot(grid=True)

####Missing values and imputation 

print(rose.isnull().sum())
##finding rows with NULL values 
null_rows=rose.isnull()
row_has_null = null_rows.any(axis=1)
rows_with_null = rose[row_has_null]
print(rows_with_null)
##Imputation using SPLINE method 
#missing values seen in 1994 Jul and Aug month 
rose['1994-01-31':'1994-12-31']
#df_missing_val.interpolate(method='spline',order=1)['1994']
#df_missing_val.interpolate(method='spline',order=2)['1994']
a=rose.interpolate(method='spline',order=3)['1994']
rose.update(a)

rose['1994-01-01':'1994-12-31']

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rose.plot(grid=True)


##yearly
sns.boxplot(x = rose.index.year,y = rose['Rose'])
plt.grid();

##monthly 
sns.boxplot(x = rose.index.month_name(),y = rose['Rose'])
plt.grid();
    
##monthly sales 
monthly_sales_across_years = pd.pivot_table(rose, values = 'Rose', columns = rose.index.month, index = rose.index.year)
monthly_sales_across_years
monthly_sales_across_years.plot()
plt.grid()
plt.legend(loc='best');


####Decomposition of TimeSeries
## additive
rose_add_decompose = seasonal_decompose(rose, model = "additive")
rose_add_decompose.plot()


trend = rose_add_decompose.trend
seasonality = rose_add_decompose.seasonal
residual = rose_add_decompose.resid

print('Trend','\n',trend.head(12),'\n')
print('Seasonality','\n',seasonality.head(12),'\n')
print('Residual','\n',residual.head(12),'\n')


####Decompose the time series multiplicatively
rose_mul_decompose = seasonal_decompose(rose, model = "multiplicative")
rose_mul_decompose.plot()


trend_m = rose_mul_decompose.trend
seasonality_m = rose_mul_decompose.seasonal
residual_m = rose_mul_decompose.resid

print('Trend','\n',trend_m.head(12),'\n')
print('Seasonality','\n',seasonality_m.head(12),'\n')
print('Residual','\n',residual_m.head(12),'\n')


####Train Test split 
rose.index.year.unique()

trainset = rose[rose.index<='1990'] 
testset = rose[rose.index>'1990']
print("train dimensions are= ",trainset.shape)
print("test dimensions are= ",testset.shape)
print(trainset.head())

####Naive Approach:  𝑦̂ 𝑡+1=𝑦𝑡 

NaiveModel_train = trainset.copy()
NaiveModel_test = testset.copy()

NaiveModel_test['naive'] = np.asarray(trainset['Rose'])[len(np.asarray(trainset['Rose']))-1]
NaiveModel_test['naive'].head()
##plot naive model
plt.plot(NaiveModel_train['Rose'], label='Train')
plt.plot(testset['Rose'], label='Test')

plt.plot(NaiveModel_test['naive'], label='Naive Forecast on Test Data')

plt.legend(loc='best')
plt.title("Naive Forecast")
plt.grid();

##metric -RMSE 
rmse_model2_test = metrics.mean_squared_error(testset['Rose'],NaiveModel_test['naive'],squared=False)
print("For Naive Model forecast on the Test Data,  RMSE is %3.3f" %(rmse_model2_test))


####Simple avg

SimpleAverage_train = trainset.copy()
SimpleAverage_test = testset.copy()

SimpleAverage_test['mean_forecast'] = trainset['Rose'].mean()
SimpleAverage_test.head()
##plot simple average 
plt.plot(SimpleAverage_train['Rose'], label='Train')
plt.plot(SimpleAverage_test['Rose'], label='Test')

plt.plot(SimpleAverage_test['mean_forecast'], label='Simple Average on ROSE Test Data')

plt.legend(loc='best')
plt.title("Simple Average Forecast on ROSE Data")
plt.grid();

##metric -RMSE 
rmse_model3_test = metrics.mean_squared_error(testset['Rose'],SimpleAverage_test['mean_forecast'],squared=False)
print("For RegressionOnTime forecast on the Test Data-ROSE ,  RMSE is %3.3f" %(rmse_model3_test))

####Moving average 
##For the moving average model, we are going to calculate rolling means 
##(or moving averages) for different intervals. 
##The best interval can be determined by the maximum accuracy (or the minimum error) over here.
MovingAverage=rose.copy()

MovingAverage['Trailing_2'] = MovingAverage['Rose'].rolling(2).mean()
MovingAverage['Trailing_4'] = MovingAverage['Rose'].rolling(4).mean()
MovingAverage['Trailing_6'] = MovingAverage['Rose'].rolling(6).mean()
MovingAverage['Trailing_9'] = MovingAverage['Rose'].rolling(9).mean()

## Plotting on the whole data
plt.plot(MovingAverage['Rose'], label='Train')
plt.plot(MovingAverage['Trailing_2'], label='2 Point Moving Average')
plt.plot(MovingAverage['Trailing_4'], label='4 Point Moving Average')
plt.plot(MovingAverage['Trailing_6'],label = '6 Point Moving Average')
plt.plot(MovingAverage['Trailing_9'],label = '9 Point Moving Average')
plt.legend(loc = 'best')
plt.grid();

#Creating train and test set 
trailing_MovingAverage_train=MovingAverage[MovingAverage.index<='1990'] 
trailing_MovingAverage_test=MovingAverage[MovingAverage.index>'1990']
print("train dimensions are= ",trailing_MovingAverage_train.shape)
print("test dimensions are= ",trailing_MovingAverage_test.shape)

## Test Data - RMSE  --> 2 point Trailing MA
rmse_model4_test_2 = metrics.mean_squared_error(testset['Rose'],trailing_MovingAverage_test['Trailing_2'],squared=False)
print("For 2 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_2))

## Test Data - RMSE --> 4 point Trailing MA
rmse_model4_test_4 = metrics.mean_squared_error(testset['Rose'],trailing_MovingAverage_test['Trailing_4'],squared=False)
print("For 4 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_4))

## Test Data - RMSE --> 6 point Trailing MA

rmse_model4_test_6 = metrics.mean_squared_error(testset['Rose'],trailing_MovingAverage_test['Trailing_6'],squared=False)
print("For 6 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_6))

## Test Data - RMSE --> 9 point Trailing MA

rmse_model4_test_9 = metrics.mean_squared_error(testset['Rose'],trailing_MovingAverage_test['Trailing_9'],squared=False)
print("For 9 point Moving Average Model forecast on the Training Data,  RMSE is %3.3f" %(rmse_model4_test_9))

##2 point moving average gives the lowest RMSE and best result

####SES

model_SES = SimpleExpSmoothing(trainset,initialization_method='estimated')

# Fitting the Simple Exponential Smoothing model and asking python to choose the optimal parameters
model_SES_autofit = model_SES.fit(optimized=True)
model_SES_autofit.params

# Using the fitted model on the training set to forecast on the test set
SES_predict = model_SES_autofit.forecast(steps=len(testset))
print("predicted vector of values",SES_predict)

## Plotting the Training data, Test data and the forecasted values

plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')

plt.plot(SES_predict, label='SimpleExpSmoothing predictions on Test Set')

plt.legend(loc='best')
plt.grid()
plt.title('Predictions from SingleExpSmoothing');


print('SES RMSE:',mean_squared_error(testset.values,SES_predict.values,squared=False))
rmse_ses=mean_squared_error(testset.values,SES_predict.values,squared=False)
print(rmse_ses)
#different way to calculate RMSE
print('SES RMSE (calculated using statsmodels):',em.rmse(testset.values,SES_predict.values)[0])


####Double Exponential Smoothing

# Initializing the Double Exponential Smoothing Model
model_DES = Holt(trainset,initialization_method='estimated')
# Fitting the model
model_DES = model_DES.fit()

print('')
print('==Holt model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_DES.params)

# Forecasting using this model for the duration of the test set
DES_predict =  model_DES.forecast(len(testset))
print("Double Exp Smoothing predictions=",DES_predict)

## Plotting the Training data, Test data and the forecasted values
plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')
plt.plot(SES_predict, label='SES on Test Set')
plt.plot(DES_predict, label='Double Exponential Smoothing predictions on Test Set')
plt.legend(loc='best')
plt.grid()
plt.title('Simple and Double Exponential Smoothing Predictions')

print('DES RMSE:',mean_squared_error(testset.values,DES_predict.values,squared=False))
rmse_des=mean_squared_error(testset.values,DES_predict.values,squared=False)


####Holt-Winters - ETS(A, A, A) - Holt Winter's linear method with additive errors

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


plt.plot(NaiveModel_test['naive'], label='Naive Forecast on Test Data')
plt.plot(SimpleAverage_test['mean_forecast'], label='Simple Average on Test Data')
plt.plot(MovingAverage['Trailing_2'], label='2 Point Moving Average')
plt.plot(SES_predict, label='Simple Exponential Smoothing predictions on Test Set')
plt.plot(DES_predict, label='Double Exponential Smoothing predictions on Test Set')
plt.plot(TES_predict, label='HoltWinters-Add On ROSE predictions on Test Set')

plt.legend(loc='best')
plt.grid()
plt.title('All Predictions');
print('TES RMSE:',mean_squared_error(testset.values,TES_predict.values,squared=False))
rmse_tes=mean_squared_error(testset.values,TES_predict.values,squared=False)

####Holt-Winters - ETS(A, A, M) - Holt Winter's linear method
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

plt.plot(NaiveModel_test['naive'], label='Naive Forecast on Test Data')
plt.plot(SimpleAverage_test['mean_forecast'], label='Simple Average on Test Data')
#plt.plot(MovingAverage['Trailing_2'], label='2 Point Moving Average')
#plt.plot(SES_predict, label='SES')
#plt.plot(DES_predict, label='DES')
plt.plot(TES_predict_am, label='HW Mul')
#plt.plot(TES_predict_am, label='HW Mul')

plt.legend(loc='best')
plt.grid()
plt.title('Simple,Double and Triple Exponential(MUL) on ROSE Smoothing Predictions');


print('TES_am RMSE:',mean_squared_error(testset.values,TES_predict_am.values,squared=False))
rmse_tes_mul=mean_squared_error(testset.values,TES_predict_am.values,squared=False)

####AugDickeyFuller Test
# Check for stationarity of the whole Time Series data.
# The Augmented Dickey-Fuller test is an unit root test which determines 
#whether there is a unit root and subsequently whether the series is non-stationary.

# The hypothesis in a simple form for the ADF test is:

# 𝐻0  : The Time Series has a unit root and is thus non-stationary.
# 𝐻1  : The Time Series does not have a unit root and is thus stationary.
# We would want the series to be stationary for building ARIMA models and thus we would want the p-value of this test to be less than the  𝛼  value.

dftest = adfuller(rose,regression='ct')
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is' ,dftest[1])
print('Number of lags used' ,dftest[2])

##p-val (0.46)! <0.05 hence failure to reject NULL hypothesis -Series is NON STATIONARY

##Let us take one level of differencing to see whether the series becomes stationary.

dftest_1 = adfuller(rose.diff().dropna(),regression='ct')
print('DF test statistic is %3.3f' %dftest_1[0])
print('DF test p-value is' ,dftest_1[1])
print('Number of lags used' ,dftest_1[2])

##Since p val =3.049 *10^-11 which <0.05 ,hence reject null hyp ,Timeseries is now stationary after FIRST differencing

####ARIMA -auto grid search 

## The following loop helps us in getting a combination of different parameters of p and q in the range of 0 and 2
## We have kept the value of d as 1 as we need to take a difference of the series to make it stationary.

import itertools
p = q = range(0, 4)
d= range(0,2)
pdq = list(itertools.product(p, d, q))
print('Examples of the parameter combinations for the Model')
for i in range(0,len(pdq)):
    print('Model: {}'.format(pdq[i]))

# Creating an empty Dataframe with column names only
ARIMA_AIC = pd.DataFrame(columns=['param', 'AIC'])
ARIMA_AIC

for param in pdq:# running a loop within the pdq parameters defined by itertools
    ARIMA_model = ARIMA(trainset['Rose'].values,order=param).fit()#fitting the ARIMA model
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
auto_ARIMA_best = ARIMA(trainset, order=(2,1,3))

results_auto_ARIMA_best = auto_ARIMA_best.fit()

print(results_auto_ARIMA_best.summary())

##diagnostics plot
results_auto_ARIMA_best.plot_diagnostics();
##prediction 
predicted_auto_ARIMA_best = results_auto_ARIMA_best.forecast(steps=len(testset))


rmse_arima_auto = mean_squared_error(testset['Rose'],predicted_auto_ARIMA_best,squared=False)
print("RMSE with grid search Arima model=",rmse_arima_auto)

####ARIMA-Manual 
##Plot the Autocorrelation and the Partial Autocorrelation function plots on the whole data
plot_acf(rose['Rose'], lags = 50)
plot_acf(rose['Rose'].diff().dropna(),lags=50,title='Differenced Data Autocorrelation')
plt.show()


plot_pacf(rose['Rose'],lags=50)
plot_pacf(rose['Rose'].diff().dropna(),lags=50,title='Differenced Data Partial Autocorrelation')
plt.show()

#Here, we have taken alpha=0.05.
##For stationary series 
#The Auto-Regressive parameter in an ARIMA model is 'p' which comes from the significant lag before which the PACF plot cuts-off at 4
#The Moving-Average parameter in an ARIMA model is 'q' which comes from the significant lag before the ACF plot cuts-off at 2
#By looking at the above plots, we can say that both the PACF and ACF plot cuts-off at lag 3 and 2 respectively.

manual_ARIMA = ARIMA(trainset['Rose'].astype('float64'), order=(4,1,2),freq='M')

results_manual_ARIMA = manual_ARIMA.fit()

print(results_manual_ARIMA.summary())

predicted_manual_ARIMA = results_manual_ARIMA.forecast(steps=len(testset))

predicted_manual_ARIMA.shape
type(predicted_manual_ARIMA)
testset.values
##Creating dataframe from Series object

rmse_manual_arima = mean_squared_error(testset.values,predicted_manual_ARIMA,squared=False)
print("Manual ARIMA model rmse = ",rmse_manual_arima)

####SARIMA -auto grid search
#Build an Automated version of a SARIMA model for which the best parameters are selected in accordance with the lowest Akaike Information Criteria 
#(AIC)

plot_acf(rose['Rose'].diff().dropna(),lags=50,title='Differenced Data Autocorrelation')
plt.show()

## We see that there can be a seasonality of  12. We will run our auto SARIMA models by setting seasonality  12.
  ##Seasonal SARIMA with seasonality=12
import itertools
p = q = range(0, 3)
d= range(0,1,2)
D = range(0,1)
pdq = list(itertools.product(p, d, q))
model_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, D, q))]
print('Examples of some parameter combinations for Model...')
for i in range(1,len(pdq)):
    print('Model: {}{}'.format(pdq[i], model_pdq[i]))


SARIMA_AIC = pd.DataFrame(columns=['param','seasonal', 'AIC'])
SARIMA_AIC

import statsmodels.api as sm

for param in pdq:
    for param_seasonal in model_pdq:
        SARIMA_model = sm.tsa.statespace.SARIMAX(trainset['Rose'].values,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            
        results_SARIMA = SARIMA_model.fit(maxiter=1000)
        print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
        SARIMA_AIC = SARIMA_AIC.append({'param':param,'seasonal':param_seasonal ,'AIC': results_SARIMA.aic}, ignore_index=True)


print(SARIMA_AIC.sort_values(by=['AIC']).head())
import statsmodels.api as sm

auto_SARIMA_best = sm.tsa.statespace.SARIMAX(trainset['Rose'].values,
                                order=(1, 0, 2),
                                seasonal_order=(2, 0, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_auto_SARIMA_best = auto_SARIMA_best.fit(maxiter=1000)
print(results_auto_SARIMA_best.summary())



results_auto_SARIMA_best.plot_diagnostics()
plt.show()

predicted_auto_SARIMA_best = results_auto_SARIMA_best.get_forecast(steps=len(testset))
predicted_auto_SARIMA_best.summary_frame(alpha=0.05).head()

rmse_auto_sarima_best = mean_squared_error(testset['Rose'],predicted_auto_SARIMA_best.predicted_mean,squared=False)
print("RMSE of SARIMA Grid Search Best model=",rmse_auto_sarima_best)


####SARIMA-Manual (observation on PACF ,ACF plots)

##For determing AR and MA non seasonal order -take ACF and PACF plots of stationary series 
plot_acf(rose['Rose'].diff().dropna(),lags=50,title='Differenced Data Autocorrelation')
plt.show()

##q=2

plot_pacf(rose['Rose'].diff().dropna(),lags=50,title='Differenced Data Partial Autocorrelation')
plt.show()
##p=4
##ARIMA order (p,d,q)=(4,1,2)

##Step 1: Do a time series plot of the data.
spark_mul_decompose = seasonal_decompose(rose, model = "multiplicative")
spark_mul_decompose.plot()

##We see there is both trend and seasonality ,hence we take seasonal differencing 
##Seasonal diff after 12 lags for sure (may be 6 also )

##Step 2 :Any necessary differencing 
(rose['Rose'].diff(12)).plot()
plt.grid();
##  :We can see slight trend,So we take a differencing of first order on the seasonally differenced series.
##If there is both trend and seasonality, apply a seasonal difference to the data and then re-evaluate the trend. If a trend remains, then take first differences
(rose['Rose'].diff(12)).diff().plot()
plt.grid();

##Now we see that there is almost no trend present in the data. Seasonality is only present in the data.

##Let us go ahead and check the stationarity of the above series before fitting the SARIMA model.

test_stationarity((trainset['Rose'].diff(12).dropna()).diff(1).dropna())
##p val=0.032
##pval <0.05 hence this TS is stationary 

##step 3 -Checking the ACF and the PACF plots for the new modified Time Series.

plot_acf((rose['Rose'].diff(6).dropna()).diff(1).dropna(),lags=30)
plot_pacf((rose['Rose'].diff(6).dropna()).diff(1).dropna(),lags=30)

##Seasonal AR order -PACF -tapers to zero after 2 lags --Each lag here is multiple of F ie 12 ;P=3
##Seasonal MA order -ACF-tapers to zero after 2 lags --Each lag here is multiple of F ie 12 ;Q=2
##Seasonal differencing D=1
##(P,D,Q,F)=(3,1,2,12)

import statsmodels.api as sm

manual_SARIMA = sm.tsa.statespace.SARIMAX(trainset['Rose'].values,
                                order=(4, 1, 2),
                                seasonal_order=(3, 1, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_manual_SARIMA = manual_SARIMA.fit(maxiter=1000)
print(results_manual_SARIMA.summary())


predicted_manual_sarima = results_manual_SARIMA.get_forecast(steps=len(testset))
predicted_manual_sarima.summary_frame(alpha=0.05).head()

##Get forecast method results in PredictionResultsWrapper object 
##this needs to converted to DF and given a date index for plotting 

pred_manual_sarima_date = predicted_manual_sarima.summary_frame(alpha=0.05).set_index(pd.date_range(start='1990-01-31',end='1995-07-31', freq='M'))

len(predicted_manual_sarima.predicted_mean)
predicted_manual_sarima.conf_int(alpha=0.05)

rmse_manual_sarima = mean_squared_error(testset['Rose'],predicted_manual_sarima.predicted_mean,squared=False)
print("RMSE of SARIMA model manual observation =",rmse_manual_sarima)


#### Result collation 
index=['Naive Approach','Simple Average Menthod','Moving Average-2Point','SES','DES','HW-A','HW-M','ARIMA-Auto(2,1,3)','ARIMA-Manual(4,1,2)','SARIMA-Auto(1,0,2)(2,0,2,12)','SARIMA-Manual(4,1,2)(3,1,2,12)']
data = pd.DataFrame({
        'RMSE':[rmse_model2_test,rmse_model3_test,rmse_model4_test_2,rmse_ses,rmse_des,rmse_tes,rmse_tes_mul,rmse_arima_auto,rmse_manual_arima,rmse_auto_sarima_best,rmse_manual_sarima]
       
        
        },index=index)
print(round(data,5))


####Plot for all models
## Plotting the Training data, Test data and the forecasted values

plt.plot(trainset, label='Train')
plt.plot(testset, label='Test')

plt.plot(NaiveModel_test['naive'], label='Naive Forecast on Test Data')
#plt.plot(SimpleAverage_test['mean_forecast'], label='Simple Average ')
plt.plot(MovingAverage['Trailing_2'], label='2 Point Moving Average')
##plt.plot(SES_predict, label='SES')
#plt.plot(DES_predict, label='DES')
plt.plot(TES_predict, label='HW Add')
#plt.plot(TES_predict_am, label='HW Mul')
plt.plot(predicted_manual_ARIMA[0] , label='ARIMA-manual(4,1,2)')
plt.plot(pred_manual_sarima_date['mean'] , label='SARIMA-manual(4,1,2)(3,1,2,12)')
plt.legend(loc='best')
plt.grid()
plt.title('All model predictions');

####Most Optimum Model on Full data 

##HW -Additive gives best results 

opt_hw_add = ExponentialSmoothing(rose,trend='additive',seasonal='additive',initialization_method='estimated')
# Fitting the model
opt_hw_ad_1= opt_hw_add.fit()

print('')
print('==Holt Winters FULL DATA model Exponential Smoothing Estimated Parameters ==')
print('')
print(opt_hw_ad_1.params)

##Find RMSE on whole of data from the optimum model
rmse2 = mean_squared_error(rose['Rose'],opt_hw_ad_1.fittedvalues,squared=False)
print('RMSE of the Full fitted HW-(ADD) Model on complete data set',rmse2)

##SARIMA opt

manual_SARIMA_opt = sm.tsa.statespace.SARIMAX(rose['Rose'].values,
                                order=(4, 1, 2),
                                seasonal_order=(3, 1, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results_manual_SARIMA_opt = manual_SARIMA_opt.fit(maxiter=1000)
print(results_manual_SARIMA_opt.summary())

rmse_manual_sarima_opt = mean_squared_error(rose['Rose'],results_manual_SARIMA_opt.fittedvalues,squared=False)
print("RMSE of SARIMA model manual observation =",rmse_manual_sarima_opt)




index=['Naive Approach','Simple Average Menthod','Moving Average-2Point','SES','DES','HW-A','HW-M','ARIMA-Auto(2,1,3)','ARIMA-Manual(4,1,2)','SARIMA-Auto(1,0,2)(2,0,2,12)','SARIMA-Manual(4,1,2)(3,1,2,12)','Optimum HW-ADD model on Full data']
data2 = pd.DataFrame({
        'RMSE':[rmse_model2_test,rmse_model3_test,rmse_model4_test_2,rmse_ses,rmse_des,rmse_tes,rmse_tes_mul,rmse_arima_auto,rmse_manual_arima,rmse_auto_sarima_best,rmse_manual_sarima,rmse2]
       
        
        },index=index)
print(round(data2,5))

# Forecasting using this model for the duration of the test set
opt_tes_predict =  opt_hw_ad_1.forecast(steps=12)
print("forecast for the next 12 months using HW FULL model",opt_tes_predict)

print(opt_hw_ad_1.resid)

####Confidence Interval for optimum model
#In the below code, we have calculated the upper and lower confidence bands at 95% confidence level
#Here we are taking the multiplier to be 1.96 as we want to plot with respect to a 95% confidence intervals.
opt_mod_ci = pd.DataFrame({'lower_CI':opt_tes_predict - 1.96*np.std(opt_hw_ad_1.resid,ddof=1),
                          'prediction':opt_tes_predict,
                          'upper_ci': opt_tes_predict+ 1.96*np.std(opt_hw_ad_1.resid,ddof=1)})
opt_mod_ci



