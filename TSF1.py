# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:40:27 2021

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
from xgboost import XGBClassifier

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
import xgboost as xgb
from xgboost import plot_importance
import pickle
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The NUMPY version is {}.'.format(np.__version__))
print('The statsmodel version is {}.'.format(statsmodels.__version__))

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


air_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 20-TimeSeriesAnalysis\AirPassenger.csv')
print(air_orig.dtypes)

##reading column as DATE  datatype
air_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 20-TimeSeriesAnalysis\AirPassenger.csv',parse_dates=['Year-Month'],index_col='Year-Month')


air=air_orig.copy()
air.drop(air.columns[air.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
air=air.dropna(how='all')
print(air.shape)

air.isnull().sum()

print(air.head(5))
print(air.dtypes)
print(air.describe())
print(air.info())

##Univariate analysis -Categorical & continuous both 
#Histograms for continuous variables
sns.distplot(air.Pax,bins=10)
##skewness
print(air['Pax'].skew())


air['1949-01-01':'1949-12-31']
    
##plotting timeseries 
air.plot()
plt.show();

#Increase the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
air.plot()
plt.show();


##decompose additively 

air_add_decomp=seasonal_decompose(air,model="additive",period=12)
air_add_decomp.plot()

##Decompose the time series multiplicatively
air_mul_decompose = seasonal_decompose(air, model = "multiplicative")
air_mul_decompose.plot()


##log transformation to make mult series into additive 
air_log=air.copy()
air_log['Pax']=np.log(air)
air_log['Pax']
air_log.plot()


#Compare with the original series
plt.subplot(2,1,1)
plt.title('Original Time Series')
plt.plot(air)

plt.subplot(2,1,2)
plt.title('Log Transformed Time Series')
plt.plot(air_log)
plt.tight_layout()

##########################################


df2 = pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 20-TimeSeriesAnalysis\daily-total-female-births.csv', parse_dates = ['Date'], index_col = 'Date')

df2.head()

df2.plot()

#Additive decomposition
df2_add_decompose = seasonal_decompose(df2, model = 'additive')

df2_add_decompose.plot()

df2_add_decompose.trend
df2_add_decompose.seasonal
df2_add_decompose.resid


#Multiplicative decomposition
df2_mul_decompose = seasonal_decompose(df2, model = 'multiplicative')
df2_mul_decompose.plot()
plt.show()

df2_mul_decompose.resid.plot()


#Let's change the monthly series to quarterly. This would require aggregation.
air_q = air.resample('Q').mean()

air_q.plot()

#Let's change the monthly series to daily. 
df1_d = air.resample('D').ffill()
df1_d['1949-02']

df1_h = air.resample('H').interpolate()

df1_h.plot()



######quiz
##reading column as DATE  datatype
age_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 20-TimeSeriesAnalysis\Normalized+Age.csv',parse_dates=['Year'],index_col='Year')


age=age_orig.copy()
age.drop(age.columns[age.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
age=age.dropna(how='all')
print(age.shape)

age.isnull().sum()

print(age.head(5))
print(age.dtypes)
print(age.describe())
print(age.info())


#Histograms for continuous variables
sns.distplot(age.Age,bins=10)
##skewness
print(age['Age'].skew())


age['1950-01-01':'1955-12-31']
    
##plotting timeseries 
age.plot()
plt.show();

#Increase the figure size
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
age.plot()
plt.show();


##decompose additively 

age_add_decomp=seasonal_decompose(age,model="additive",period=12)
age_add_decomp.plot()

##Decompose the time series multiplicatively
age_mul_decompose = seasonal_decompose(age, model = "multiplicative")
age_mul_decompose.plot()


##log transformation to make mult series into additive 
age_log=age.copy()
age_log['Age']=np.log(age)
age_log['Age']
age_log.plot()


#Compare with the original series
plt.subplot(2,1,1)
plt.title('Original Time Series')
plt.plot(age)

plt.subplot(2,1,2)
plt.title('trend Time Series')
plt.plot(t)
plt.tight_layout()

t=age_add_decomp.trend
t.plot()
age_add_decomp.seasonal
age_add_decomp.resid

#Let's change the yearly series to quarterly. This would require aggregation.
age_q = age.resample('Q').ffill()

age_q.plot()

#Let's change the monthly series to daily. 
df1_d = air.resample('D').ffill()
df1_d['1949-02']

df1_h = air.resample('H').interpolate()

df1_h.plot()











