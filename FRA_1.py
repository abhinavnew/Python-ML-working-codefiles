# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:55:41 2021

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
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler ##For zscore scaling 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster ##Hierachical clustering
from sklearn.metrics import silhouette_samples, silhouette_score  ##kmeans sil score
from sklearn.tree import DecisionTreeClassifier   ##DT for CART
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve,recall_score
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


####Read Data
default_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 32-33 FRA\Consumer Credit Risk\Default.csv')

default=default_orig.copy()
default.drop(default.columns[default.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
default=default.dropna(how='all')
print(default.shape)
print(default.columns)

default.isnull().sum()

print(default.head(5))
print(default.dtypes)
print(default.describe())
print(default.info())

####Univariate analysis -Categorical & continuous both 

print(default['student'].unique())

print(default['default'].unique())

#Histograms for continuous variables

sns.distplot(default.balance,bins=10)
sns.distplot(default.income,bins=10)
##skewness
print(default['balance'].skew())
print(default['income'].skew())


#Count bar plots for categorical variables 
default['student'].value_counts().plot(kind='bar')
default['default'].value_counts().plot(kind='bar')

####BiVariate analysis 

##Categorical predictor with Target
sns.countplot(default['student'],hue=default['default'])

##Continuous predictor with Target
sns.boxplot(default['income'],default['default'])

##pair plot
sns.pairplot(default,hue='default',diag_kind='kde')
plt.show()

##Heatmap
plt.figure(figsize=(12,7))
sns.heatmap(default.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(default.corr()) )
plt.show()

##are the categorical independent variables having any relationship (statistically)with Categorical Target ?
#H0 -both are independent Ha-Both are not independent


ct=pd.crosstab(default.student,default.default)
ct.reset_index(level=0,inplace=True)

chi2, pval, dof, exp_freq = chi2_contingency(ct, correction = False)

print(pval)
## 1.8210186961067486e-107  pval <0.05 reject null hypothesis 


##Duplicate analysis 
dups=default.duplicated()
print("The num of duplicate records are =",dups.sum())
default[dups]
default.drop_duplicates(inplace=True)

print(default.shape)



##Shapiro-wilk test for normality 
# 𝐻0: age follows a normal distribution 
# against 
# 𝐻𝑎: age does not follow a normal distribution
# if p-val<0.05 then Null hyp is rejected 

w, p_value = stats.shapiro(default['income']) 
print("W = {}".format(w), "p_value = {}".format(p_value))
##p-val too small ,we reject H0 ,income does NOT follow normality ,we need to apply trnsformation
# BOX-COX transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(default['income'])

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
  
# plotting the original data(non-normal) and 
# fitted data (normal)
sns.distplot(default['income'], hist = False, kde = True,
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
default.boxplot()
plt.xticks(rotation=90)
print('Shape before Outliers Treatment',default.shape)


ll1,ul1=getupperlower_outlier(default['balance'])
out1=default.loc[(default['balance']>ul1)|(default['balance']<ll1),]
print(out1.shape)

##Missing and Bad data 
print(default.isnull().sum())
default.info()

##Checking for any columns having constant values 
zero_var_col=return_constant_columns(default)


##Scaling not required for Logistic Reg,NB,Bagging (RF,Bagging tree based ),Boosting (XGB,GBM tree based)


##Scaling impacts -KNN,Kmeans,NN,LDA 
##We need to perform Feature Scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression, Neural Network) 
##and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points.

##Multicollinearity -not enough to drop columns
plt.figure(figsize=(12,7))
sns.heatmap(default.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(default.corr()) )
plt.show()

##Encoding 
##For gender we can code randomly -corresponding to categorical codes
default['student'] = pd.Categorical(default['student']).codes 
default['default'] = pd.Categorical(default['default']).codes 
#ordered -Ordinal coding 
encobj_desig=OrdinalEncoder(categories=[['Labour','Conservative']])
default['vote']=encobj_desig.fit_transform(default[["vote"]])    
default['vote'].value_counts().plot(kind='bar')  
default['vote']=(default['vote']).astype(int)


##train and validate split
all_labels=default['default']
all_ind_ds=default.drop(["default"],axis=1)

# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels =train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=400)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

##Baseline model 
##predicts the dominant class of 0 for all observations 
##Baseline accuracy on Overall set =9667/(9667+333) on  TESTSET =2899/(2899+101)=96.7%  this we need to beat atleast



# Fit the Logistic Regression model
model_logis = LogisticRegression(solver='newton-cg',max_iter=10000,verbose=True,n_jobs=2)
model_logis.fit(trainset,train_labels)
pred_test=model_logis.predict(testset)
pred_train=model_logis.predict(trainset) 

print(confusion_matrix(train_labels,pred_train))
log_acc_train=accuracy_score(train_labels,pred_train)
print(accuracy_score(train_labels,pred_train)) 

probs_train = model_logis.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train = probs_train[:, 1]
# calculate AUC
auc_train = roc_auc_score(train_labels, probs_train)
print('AUC: %.3f' % auc_train)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(train_labels, probs_train)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr)
##Test set
print(confusion_matrix(test_labels,pred_test))
log_acc_test=accuracy_score(test_labels,pred_test)
print(log_acc_test)
print(accuracy_score(test_labels,pred_test))
 
probs_test = model_logis.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
# calculate AUC
auc_test = roc_auc_score(test_labels, probs_test)
print('AUC: %.3f' % auc_test)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(test_labels, probs_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr)

import statsmodels.formula.api as SM

frmul='default~student+balance+income'


smtrainset=pd.concat([trainset,train_labels],axis=1)
smtestset=pd.concat([testset,test_labels],axis=1)

mod_log2=SM.logit(formula=frmul,data=smtrainset).fit()
mod_log2.summary()

#predicting on trainset
y_pred_train = np.where(mod_log2.predict(smtrainset) > 0.5, 1, 0)


print(confusion_matrix(train_labels, y_pred_train))

print(classification_report(train_labels, y_pred_train))

#predicting on testset

y_pred_test = np.where(mod_log2.predict(smtestset) > 0.5, 1, 0)


print(confusion_matrix(test_labels, y_pred_test))

print(classification_report(test_labels, y_pred_test))


####SMOTE for trainset only
sm = SMOTE(random_state=33, sampling_strategy = 0.75)
X_res, y_res = sm.fit_resample(trainset, train_labels)

default_smote_train = pd.concat([X_res, y_res], axis = 1)


####Logistic regression on balanced data 
mod_log3=SM.logit(formula=frmul,data=default_smote_train).fit()
mod_log3.summary()

y_pred_train_smote = np.where(mod_log3.predict(default_smote_train) > 0.5, 1, 0)
print(confusion_matrix(y_res, y_pred_train_smote))
print(classification_report(y_res, y_pred_train_smote))
print(recall_score(y_res, y_pred_train_smote))


y_pred_test_smote = np.where(mod_log3.predict(smtestset) > 0.5, 1, 0)


print(confusion_matrix(test_labels, y_pred_test_smote))
print(classification_report(test_labels, y_pred_test_smote))
print(recall_score(test_labels, y_pred_test_smote))











