# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:04:46 2021

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
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))

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


xls=pd.ExcelFile(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\MRA Project\Sales_Data.xlsx')

retail_orig=pd.read_excel(xls,'Online Retail')
data_dict=pd.read_excel(xls,'Data Description')
retail=retail_orig.copy()
retail.drop(retail.columns[retail.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
retail=retail.dropna(how='all')
print(retail.shape)

retail.isnull().sum()

print(retail.head(5))
print(retail.dtypes)
print(retail.describe())
print(retail.info())

##Univariate analysis -Categorical & continuous both 
print(retail['economic.cond.national'].unique())
print(retail['economic.cond.household'].unique())
print(retail['Blair'].unique())
print(retail['Hague'].unique())
print(retail['Europe'].unique())

print(retail['political.knowledge'].unique())
print(retail['gender'].unique())
##Continuous -uniq values
print(retail['age'].unique())

#Histograms for continuous variables
sns.distplot(retail.age,bins=10)
##skewness
print(retail['age'].skew())


#Count bar plots for categorical variables 
retail['economic.cond.national'].value_counts().plot(kind='bar')
retail['economic.cond.household'].value_counts().plot(kind='bar')
retail['Blair'].value_counts().plot(kind='bar')
retail['Hague'].value_counts().plot(kind='bar')


retail['Europe'].value_counts().plot(kind='bar')
retail['political.knowledge'].value_counts().plot(kind='bar')
retail['gender'].value_counts().plot(kind='bar')


##target
retail['vote'].value_counts().plot(kind='bar')

#Histograms for continuous variables
sns.distplot(retail.age,bins=10)



##Skewness check for continuous variables 
print(retail['age'].skew())


##BiVariate analysis 

##Categorical predictor with Target
sns.countplot(retail['economic.cond.national'],hue=retail['vote'])
sns.countplot(retail['economic.cond.household'],hue=retail['vote'])
sns.countplot(retail['Blair'],hue=retail['vote'])
sns.countplot(retail['Hague'],hue=retail['vote'])
sns.countplot(retail['Europe'],hue=retail['vote'])
sns.countplot(retail['political.knowledge'],hue=retail['vote'])
sns.countplot(retail['gender'],hue=retail['vote'])
##Continuous predictor with Target
sns.boxplot(retail['vote'],retail['age'])

retail.groupby(['vote'],as_index=False).agg({'age':'median'})



##pair plot
sns.pairplot(retail,hue='vote',diag_kind='kde')
plt.show()

##Heatmap
plt.figure(figsize=(12,7))
sns.heatmap(retail.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(retail.corr()) )
plt.show()


##are the categorical independent variables having any relationship (statistically)with Categorical Target ?
#H0 -both are independent Ha-Both are not independent


ct=pd.crosstab(retail.Blair,retail.vote)
ct.reset_index(level=0,inplace=True)

chi2, pval, dof, exp_freq = chi2_contingency(ct, correction = False)

print(pval)
## 1.8210186961067486e-107  pval <0.05 reject null hypothesis 


##Duplicate analysis 
dups=retail.duplicated()
print("The num of duplicate records are =",dups.sum())
retail[dups]
retail.drop_duplicates(inplace=True)

print(retail.shape)

##Shapiro-wilk test for normality 
# 𝐻0: age follows a normal distribution 
# against 
# 𝐻𝑎: age does not follow a normal distribution
# if p-val<0.05 then Null hyp is rejected 

w, p_value = stats.shapiro(retail['age']) 
print("W = {}".format(w), "p_value = {}".format(p_value))
##p-val too small ,we reject H0 ,age does NOT follow normality ,we need to apply trnsformation
# BOX-COX transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(retail['age'])

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
  
# plotting the original data(non-normal) and 
# fitted data (normal)
sns.distplot(retail['age'], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Non-Normal", color ="green", ax = ax[0])
  
sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Normal", color ="green", ax = ax[1])
  
# adding legends to the subplots
plt.legend(loc = "upper right")
  
# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
  
print(f"Lambda value used for Transformation: {fitted_lambda}")

##outlier analysis
retail.boxplot()
plt.xticks(rotation=90)
print('Shape before Outliers Treatment',retail.shape)

ll1,ul1=getupperlower_outlier(retail['economic.cond.national'])
out1=retail.loc[(retail['economic.cond.national']>ul1)|(retail['economic.cond.national']<ll1),]
print(out1.shape)

ll2,ul2=getupperlower_outlier(retail['economic.cond.household'])
out2=retail.loc[(retail['economic.cond.household']>ul2)|(retail['economic.cond.household']<ll2),]
print(out2.shape)



##Missing and Bad data 
print(retail.isnull().sum())
retail.info()

##Checking for any columns having constant values 



zero_var_col=return_constant_columns(retail)

##Checking for any bad values (character/string columns)

# def select_badvalues_rows(dframe,cat_colnames,bad_vals):
#     smalldfs=[]
#     pat='|'.join(['({})'.format(re.escape(c))for c in bad_vals ])
#     for i in cat_colnames :           
#         smalldfs.append(dframe.loc[dframe[i].str.contains(pat)])
#         print("For column name =",i)
#         print(smalldfs)
#     largedf=pd.concat(smalldfs,ignore_index=True)    
#     return largedf

#get any bad values
bad_values = ['\?','NA','\@','None','NaN','Nan','nan','Missing','-99','-999','\??','\???'] 
elect=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\ML_Ensemble_TextAnalytics_project\elect.csv')

cat_col,num_col=get_num_cat_colnames(elect)
c=select_badvalues_rows(elect,cat_col,bad_values)


##Scaling not required for Logistic Reg,NB,Bagging (RF,Bagging tree based ),Boosting (XGB,GBM tree based)
##Scaling impacts -KNN,Kmeans,NN,LDA 

##Multicollinearity -not enough to drop columns
plt.figure(figsize=(12,7))
sns.heatmap(retail.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(retail.corr()) )
plt.show()


##Encoding 
##For gender we can code randomly -corresponding to categorical codes
retail['gender'] = pd.Categorical(retail['gender']).codes 
##For vote -we want to code conservative=1 and hence will use ordinal encoder method
encobj_desig=OrdinalEncoder(categories=[['Labour','Conservative']])
retail['vote']=encobj_desig.fit_transform(retail[["vote"]])    
retail['vote'].value_counts().plot(kind='bar')  
retail['vote']=(retail['vote']).astype(int)
       

##train and validate split
all_labels=retail['vote']
all_ind_ds=retail.drop(["vote"],axis=1)

# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels =train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=400)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)



e=retail.copy()
a=fitted_data.reshape(-1,1)
e['age']=a

##train and validate split with age transformed for NB (box-cox)
all_l=e['vote']
all_i=e.drop(["vote"],axis=1)

# splitting data into training and test set for independent attributes
trset,teset, tr_labels, te_labels =train_test_split(all_i, all_l, test_size=.30, random_state=400)
print(trset.shape)
print(tr_labels.shape)
print(teset.shape)
print(te_labels.shape)


##Baseline model 
##predicts the dominant class of 0 for all observations 327/(327+129)
##Baseline accuracy on TESTSET =71.7% this we need to beat atleast
###########################################################################
##Naive Bayes 

nb_obj=GaussianNB()
nb_model=nb_obj.fit(trset,tr_labels)

pred_train_nb=nb_model.predict(trset)
pred_test_nb=nb_model.predict(teset)
