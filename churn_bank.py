# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 08:56:44 2021

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
import csv
from scipy.stats import chi2_contingency ## for chi sq test 
from scipy.stats import   ttest_1samp,ttest_ind ##For t test
from scipy.stats import variation  
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from sklearn.preprocessing import StandardScaler ##For zscore scaling 
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier   ##DT for CART
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
 ##for clustering
from sklearn.cluster import KMeans  ## for k means
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
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

bad_values = ['\?','NA','\@','None','NaN','Nan','nan','Missing','-99','-999'] 

def select_badvalues_rows(dframe,cat_colnames,bad_vals):
    smalldfs=[]
    for i in cat_colnames :           
        smalldfs.append(dframe.loc[dframe[i].str.contains('|'.join(bad_values))==True])
        print("For column name =",i)
    print(type(smalldfs))
    largedf=pd.concat(smalldfs,ignore_index=True)    
    return largedf

c=select_badvalues_rows(df1,cat_col,bad_values)
cat_col,num_col=get_num_cat_colnames(df1)

def replace_badvalues_withNan(dframe,cat_colnames,bad_vals):
    for i in cat_colnames :           
        dframe[i].str.replace('|'.join(bad_values,np.NaN))
        print("For column name =",i)

# c=select_badvalues_rows(churn,cat_col,bad_values)
# c.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week13-NeuralNet\c.csv')

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


def treat_outlier_5_95(x):
    # taking 5,25,75 percentile of column
    q5= np.percentile(x,5)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    dt=np.percentile(x,95)
    #calculationg IQR range
    IQR=q75-q25
    print("Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: dt if y > upper_bound else y).apply(lambda y: q5 if y < lower_bound else y)

 
 

        
c=badvalues_rows(churn,cat_col,bad_values)
c=pd.DataFrame(columns=churn.columns)
c.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week13-NeuralNet\c.csv')
np.savetxt("c.csv", c, delimiter =",",fmt ='% s')
    
def get_num_cat_colnames(dframe):
        hl_cat=[]
        hl_num=[]
        for i in dframe.columns :
            if dframe[i].dtype=="object":
                hl_cat.append(i)
            else :
                hl_num.append(i)
        print(hl_cat)
        print(hl_num)
        return hl_cat,hl_num
    
    
 ##zero variance or constant features
def drop_constant_columns_4(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    keep_columns = dataframe.columns[dataframe.nunique()>1]
    a=dataframe.loc[:,keep_columns].copy()
    print(a.shape)
    return a
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)

##Basic EDA ::Columns dtypes,shape,unique values 
##Basic EDA ::Which Features are Categorical (Character)Or Categorical (Numerical),Which are continuous
##correlated columns,same value or Zero variance columns;; DOM -SMET (Duplicates,outliers,missing-Scaling mlticollinr,encoding,transformation)


churn_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week13-NeuralNet\\Churn_Modelling-1.csv')
churn=churn_orig.copy()


##Duplicates
dups=churn.duplicated()
print("The no of dupicates are :",dups.sum())

##Outliers
churn.boxplot()
plt.xticks(rotation=90);

##Missing values

churn.isnull().sum()
##Finding cols with special charac
cat_col,num_col=get_num_cat_colnames(churn)
print(cat_col)
print(num_col)

##replacing bad values with NULL

churn['Geography'].replace('?',np.NaN,inplace=True)
churn['Gender'].replace('@',np.NaN,inplace=True)
churn['Age'].replace('?',np.NaN,inplace=True)
churn['Balance'].replace('?',np.NaN,inplace=True)

churn.isnull().sum()
#imputing outlier continuous columns only with median 
for i in churn[['Credit Score']]:
    median = churn[i].median()
    churn[i].replace(np.nan, median, inplace= True)
#imputing non outliers continuous columns only with mean
for column in churn[['Tenure', 'Estimated Salary']]:
    mean = churn[column].mean()
    churn[column] = churn[column].fillna(mean)

#imputing categoricals columns only with mode
for column in churn[['Geography','Gender','Has CrCard','Is Active Member']]:
    if churn[column].dtype == 'object':
        churn[column] = pd.Categorical(churn[column]).codes

for column in churn[['Tenure', 'Estimated Salary']]:
    mean = churn[column].mean()
    churn[column] = churn[column].fillna(mean)





df['col1']=df['col1'].astype('float64')
##check and replace with col3 if condition true else let it be col2
df['col2']=np.where(df['col2']==-1,df['col3'],df['col2'])

##impute with mean when bad val -1
df['col5'].replace(-1,df['col5'].mean(),inplace=True)

##impute with mean/median for missing values (if no outliers then safe)
df.col6.fillna(df.col6.mean(),inplace=True)

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df_numericcols=pd.DataFrame(imputer.fit_transform(df_numericcols),columns=df_numericcols.columns)

#imputing continuous columns only with median 
for i in usheart[['age', 'cigsPerDay', 'tot cholesterol', 'Systolic BP', 'Diastolic BP','BMI','heartRate','glucose']]:
    median = usheart[i].median()
    usheart[i].replace(np.nan, median, inplace= True)
    
    usheart.isnull().sum()



