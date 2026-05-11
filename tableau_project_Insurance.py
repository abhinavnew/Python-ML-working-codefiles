# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:45:04 2021

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
import os
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
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
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_auc_score,roc_curve
 ##for clustering
import time
from datetime import timedelta
from sklearn.cluster import KMeans  ## for k means
#from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import make_scorer
sns.set(color_codes=True) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import plot_importance
import pickle
from sklearn.impute import KNNImputer
import feature_engine
from feature_engine.wrappers import SklearnTransformerWrapper


def getupperlower_outlier(col):
    sorted(col)
    if col.isnull().values.any()==True:
        Q1,Q3=np.nanpercentile(col,[25,75])
        count=col.isnull().sum()
        print("There are NaN values in the column,which we will ignore-No of NaNs=",count)
    else:
        Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    print("Inside function:Lower range = ",lower_range)
    print("Inside function:Upper range = ",upper_range)
    return lower_range, upper_range



def treat_outlier_5_95(x):
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
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 30)

pd.set_option("display.max_colwidth",30)
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

####Read file
cars_orig=pd.read_excel(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 23-25-Tableau\InsuranceClaims.xlsx')
cars=cars_orig.copy()
print(cars.shape)

####Droping all rows with NA/NULL values
cars=cars.dropna(how='all')

#ID-unique
#Age-continuous 
#recently_upskilled-Categorical binary 
#region-Categorical -text 
#RemoteWork-Categorical binary
#%salary Increment -continuous 
#Office hours -continuous 
#stockoption level-Categorical binary 
#College tier-Categorical -text 
#YearsSinceLastPromotion-Continuous -numerical
#HighestEducation-Categorical -text object 
#Businesstravel-Categorical -text object 
#Joblevel-Categorical -numerical encoded
#Marital status -Categorical -text object 
#Monthly salary -Continuous 
#yof joining -Categorical
#TotalWorkExp-continuous -numerical
#Gender -Categorical -text object 
#YearswithCurrManager-continuous 
#DistanceFrmHome-Continuous 
#NumCompaniesWorked -continuous
#TrainingingTimesLastYear -continuous 
#Dept-Categorical -text 
#Envsat1-categorical -encoded already 
#EnvSat2-Categorical -encoded already 
#Jobsat1 -encoded
#job sat2 -categorical -encoded
#Job invol1-Cat-encoded 
#Job involv2 -cat -encoded 
#Performancerating1 -Cat-encoded
#Performrating2-Cat-encoded
#EmpTurnover-target

print(cars.shape)
print(cars.columns)
print(cars.head(5))
print(cars.dtypes)
print(cars.describe())
print(cars.info())


####outlier analysis
cars.boxplot()
plt.xticks(rotation=90);
sns.boxplot(cars['CAR_AGE'])
##No outliers in car age ,we can replace negative value with NA and then with mean 
##impute with mean when bad val -3
cars['CAR_AGE'].replace(-3,cars['CAR_AGE'].mean(),inplace=True)

#Binning the car age
cars['CAR_AGE'].describe()
bins = [-np.inf,1,8,12,np.inf]
label =['New Cars','Fairly_New Cars','Fairly_Old Cars','Very Old Cars']
cars['Car_Age_Slabs'] = pd.cut(cars['CAR_AGE'],bins ,right=True,labels=label)
print(cars.head(20))

#Binning the Income 

bins = [-np.inf,26125,51121,81002.5,np.inf]
label =['Low Income','Lower_middle Income','Upper_middle Income','High Income']
cars['Income_Slabs'] = pd.cut(cars['INCOME'],bins ,right=True,labels=label)
print(cars.head(20))












