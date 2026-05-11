# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:07:23 2021

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
from sklearn.impute import KNNImputer
import feature_engine
from feature_engine.wrappers import SklearnTransformerWrapper

from sklearn.feature_selection import RFE
import statsmodels.formula.api as SM
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
credit_orig=pd.read_excel(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 32-33 FRA\Milestone1\Company_Data2015-1.xlsx')
dt_dic=pd.read_excel(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 32-33 FRA\Milestone1\Data_Dictionary.xlsx')
credit=credit_orig.copy()
credit.drop(credit.columns[credit.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
credit=credit.dropna(how='all')
print(credit.shape)
print(credit.columns)

credit.isnull().sum()

print(credit.head(5))
print(credit.dtypes)
print(credit.describe())
print(credit.info())

####Univariate analysis -Categorical & continuous both 

print(credit['Co_Code'].nunique())
print(credit['Co_Name'].nunique())

#Histograms for continuous variables

sns.distplot(credit['Networth Next Year'],bins=2)
sns.distplot(credit['Networth'],bins=2)
sns.distplot(credit['PBIT'],bins=3)
sns.distplot(credit['Adjusted PAT'],bins=2)
sns.distplot(credit['CP'],bins=10)
##skewness
print(credit['Networth Next Year'].skew())
print(credit['Equity Paid Up'].skew())

####creating target variable -default
credit['default']=np.where(credit['Networth Next Year']<=0,1,0)


####Bi variate analysis with target variable 

##Continuous predictor with Target



####Duplicate analysis 
dups=credit.duplicated()
print("The num of duplicate records are =",dups.sum())
credit[dups]
credit.drop_duplicates(inplace=True)

print(credit.shape)

#### Dropping unique colums Company Id and Name as they will not provide any extra info to the model stage 
##Also KNN Imputer requires that all inputs are numeric 
credit_num=credit.drop(['Co_Code', 'Co_Name'],axis=1)

####Scaling though not requred for LR but required for KNN imputation 

knntrain=credit_num.drop(['default'],axis=1)
knnlabels=credit_num['default']

scaler = StandardScaler()
scaled_predictors = pd.DataFrame(scaler.fit_transform(knntrain), columns = knntrain.columns)

knn_scaled=pd.concat([scaled_predictors, knnlabels], axis = 1)

####Missing value imputation using KNN IMPUTER 

# print(knntrain.shape)
# print(knnlabels.shape)
# print(knntrain.shape)
# print(len(knnlabels))

imputer = KNNImputer(n_neighbors=10)
credit_imp = pd.DataFrame(imputer.fit_transform(knn_scaled), columns = knn_scaled.columns)


print("Running KNN imputation now ....")
print("null values in original Scaled  dataset ",knn_scaled.isnull().sum())
print("null values after KNN imputed dataset ",credit_imp.isnull().sum())


####Outlier analysis
print("Outlier analysis before any treatment")
credit_imp.boxplot();
plt.xticks(rotation=90);



# ll1,ul1=getupperlower_outlier(credit['Networth'])
# out1=credit.loc[(credit['Networth']>ul1)|(credit['Networth']<ll1),]
# print(out1.shape)


####getting cat and num cols 
catcol,numcol=get_num_cat_colnames(credit_imp)


#### outlier treatment -pure winsorization 
numcol.remove('default')  #### we dont want to treat target column
for j in numcol:
    credit_imp[j]=treat_outlier_ul_ll_winsor(credit_imp[j])  
    print("Winsorization complete for column=",j)


print('Shape after Outliers Treatment',credit_imp.shape)
credit_imp.boxplot();
plt.xticks(rotation=90);
####Scaling -Not required for Logistic regression 
####Multicollinearity 

plt.figure(figsize = (12,8))
cor_matrix = credit_imp.drop('default', axis = 1).corr()
sns.heatmap(cor_matrix, cmap = 'rainbow', vmin = -1, vmax= 1)

##some very correlated values are seen --->?????


##train and validate split
all_labels=credit_imp['default']
all_labels=all_labels.astype(int)
all_ind_ds=credit_imp.drop(['default'],axis=1)

all_ind_ds.columns=all_ind_ds.columns.str.replace('%','Pct')
all_ind_ds.columns=all_ind_ds.columns.str.replace('(','')
all_ind_ds.columns=all_ind_ds.columns.str.replace(')','')
all_ind_ds.columns=all_ind_ds.columns.str.replace('[','')
all_ind_ds.columns=all_ind_ds.columns.str.replace(']','')
all_ind_ds.columns=all_ind_ds.columns.str.replace('.','')
all_ind_ds.columns=all_ind_ds.columns.str.replace(' ','_')
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels =train_test_split(all_ind_ds, all_labels, test_size=.33, random_state=42)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

####For modeling we will use Logistic Regression with recursive feature elimination and show rank 

LogR = LogisticRegression()

selector = RFE(estimator = LogR, n_features_to_select=15, step=1)

selector = selector.fit(trainset,train_labels)

selector.n_features_

selector.ranking_

df = pd.DataFrame({'Feature': trainset.columns, 'Rank': selector.ranking_})
df[df['Rank'] == 1]

# #Statsmodel requires the labelled data, therefore, concatinating the y label to the train set.
# stats_train = pd.concat([trainset,train_labels], axis=1)
# stats_test = pd.concat([testset,test_labels], axis=1)
# frmul='default~ Networth_Next_Year+ Networth+ PBIDT'
# smmodel = SM.logit(formula =frmul,data=stats_train).fit()

pred_train = selector.predict(trainset)
pred_test =selector.predict(testset)

print(confusion_matrix(train_labels, pred_train))

print(classification_report(train_labels, pred_train))


print(recall_score(train_labels, pred_train))
print(accuracy_score(train_labels, pred_train))

##on testset
print(confusion_matrix(test_labels, pred_test ))
print(classification_report(test_labels, pred_test ))
print(recall_score(test_labels, pred_test ))

probs4 = selector.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs4 = probs4[:, 1]
# calculate AUC
auc4 = roc_auc_score(train_labels, probs4)
print('AUC on Trainset for LR  model = %.3f' % auc4)

# ##optimal threshold 

for j in np.arange(0.1,1,0.1):
    custom_prob = j #defining the cut-off value of our choice
    custom_cutoff_data=[]#defining an empty list
    for i in range(0,len(train_labels)):#defining a loop for the length of the train data
        if np.array(probs4)[i] > custom_prob:#issuing a condition for our probability values to be 
            #greater than the custom cutoff value
            a=1#if the probability values are greater than the custom cutoff then the value should be 1
        else:
            a=0#if the probability values are less than the custom cutoff then the value should be 0
        custom_cutoff_data.append(a)#adding either 1 or 0 based on the condition to the end of the list defined by us
    print(round(custom_prob,3),round(metrics.recall_score(train_labels,custom_cutoff_data),4))


probs3 = selector.predict_proba(testset)

data_pred_custom_cutoff_validate=[]
for i in range(0,len(probs3)):
    if np.array(probs3)[i]>0.2:
        a=1
    else:
        a=0
    data_pred_custom_cutoff_validate.append(a)



####SMOTE as recall is quite bad 


sm = SMOTE(random_state=33)
X_res, y_res = sm.fit_resample(trainset, train_labels)

selector_smote = selector.fit(X_res, y_res)

selector_smote.n_features_

pred_train_smote = selector_smote.predict(X_res)
pred_test_smote = selector_smote.predict(testset)

print(classification_report(y_res, pred_train_smote))

print(classification_report(test_labels, pred_test_smote))

print(recall_score(test_labels, pred_test_smote))



