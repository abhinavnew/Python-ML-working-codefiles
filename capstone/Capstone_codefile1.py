# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:52:58 2021

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


xls=pd.ExcelFile(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\ML_Ensemble_TextAnalytics_project\Election_Data.xlsx')

election_orig=pd.read_excel(xls,'Election_Dataset_Two Classes')
data_dict=pd.read_excel(xls,'Sheet1')

####Read Data
fullexcel=pd.ExcelFile(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\capstone options\CC_EDTH_02_Customer Churn (1)\Customer Churn Data.xlsx')
capchurn_orig=pd.read_excel(fullexcel,'Data for DSBA')
data_dict=pd.read_excel(fullexcel,'Meta Data')


capchurn=capchurn_orig.copy()
capchurn.drop(capchurn.columns[capchurn.columns.str.contains('unnamed',case=False)],axis=1,inplace=True)
##Droping all rows with NA/NULL values
capchurn=capchurn.dropna(how='all')
print(capchurn.shape)
print(capchurn.columns)

capchurn.isnull().sum()

print(capchurn.head(5))
print(capchurn.dtypes)
print(capchurn.describe())
print(capchurn.info())

####Replacing bad values with NAN
capchurn['Tenure'].replace('#',np.NaN,inplace=True)
capchurn['Tenure']=capchurn['Tenure'].astype('float64')

capchurn['Account_user_count'].replace('@',np.NaN,inplace=True)
capchurn['Account_user_count']=capchurn['Account_user_count'].astype('float64')


capchurn['rev_per_month'].replace('+',np.NaN,inplace=True)
capchurn['rev_per_month']=capchurn['rev_per_month'].astype('float64')


capchurn['rev_growth_yoy'].replace('$',np.NaN,inplace=True)
capchurn['rev_growth_yoy']=capchurn['rev_growth_yoy'].astype('float64')

capchurn['coupon_used_for_payment'].replace('#',np.NaN,inplace=True)
capchurn['coupon_used_for_payment'].replace('$',np.NaN,inplace=True)
capchurn['coupon_used_for_payment'].replace('*',np.NaN,inplace=True)

capchurn['Day_Since_CC_connect'].replace('$',np.NaN,inplace=True)

capchurn['cashback'].replace('$',np.NaN,inplace=True)

capchurn['Login_device'].replace('&&&&',np.NaN,inplace=True)


####Univariate analysis -Categorical & continuous both 

print(capchurn['Tenure'].unique())
print(capchurn['Gender'].unique())
print(capchurn['Account_user_count'].unique())
print(capchurn['account_segment'].unique())
print(capchurn['Marital_Status'].unique())
print(capchurn['rev_per_month'].unique())
print(capchurn['rev_growth_yoy'].unique())
print(capchurn['coupon_used_for_payment'].unique())
print(capchurn['Day_Since_CC_connect'].unique())
print(capchurn['cashback'].unique())
print(capchurn['Login_device'].unique())


#Histograms for continuous variables

sns.distplot(capchurn['Tenure'])
sns.distplot(capchurn['CC_Contacted_LY'])
sns.distplot(capchurn['rev_per_month'])
sns.distplot(capchurn['rev_growth_yoy'])
sns.distplot(capchurn['Day_Since_CC_connect'])

sns.distplot(capchurn['cashback'])
sns.displot(capchurn['rev_per_month'])

# tenure 
# CC_Contacted_LY
# rev_per_month

# rev_growth_yoy
# Day_Since_CC_connect
# cashback
# rev_per_month


#### categorical variables 
#Count bar plots for categorical variables 

capchurn['City_Tier'].value_counts().plot(kind='bar')
capchurn['Payment'].value_counts().plot(kind='bar')
capchurn['Gender'].value_counts().plot(kind='bar')

# churn
# City_Tier
# Payment
# Gender
# Service_Score

# Account_user_count
# account_segment
# CC_Agent_Score
# Marital_Status
# Complain_ly

# coupon_used_for_payment
# Login_device



####Bi variate analysis with target variable 

#### Heat map with masking 


plt.figure(figsize=(12,7))
sns.heatmap(capchurn.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(capchurn.corr()) )
plt.show()


##Missing values Imputation -Mode/median
capchurn.isnull().sum()

catcol,numcol=get_num_cat_colnames(capchurn)
#imputing continuous columns only with median 
for i in numcol:
    median = capchurn[i].median()
    capchurn[i].replace(np.nan, median, inplace= True)
    
    capchurn.isnull().sum()

for j in capchurn[['Payment', 'Gender', 'account_segment', 'Marital_Status', 'Login_device']]:
    mode=capchurn[j].mode()
    print("mode of this column",mode[0])
    capchurn[j].replace(np.nan,mode[0],inplace=True)
    
capchurn.isnull().sum()
    
    
    
####Outlier analysis post missing val imputatio
    
# capchurn.boxplot()
# plt.xticks(rotation=90);


capchurn.boxplot(column=['Tenure']);

capchurn.boxplot(column=['CC_Contacted_LY']);

capchurn.boxplot(column=['Day_Since_CC_connect']);

capchurn.boxplot(column=['rev_per_month']);

capchurn.boxplot(column=['coupon_used_for_payment']);

capchurn.boxplot(column=['cashback']);

for k in capchurn[['Tenure','CC_Contacted_LY','Day_Since_CC_connect','rev_per_month','coupon_used_for_payment','cashback']]:
    capchurn[k]=treat_outlier_ul_ll_winsor(capchurn[k])  
    print("Winsorization complete for column=",k)
    
# capchurn.boxplot();
# plt.xticks(rotation=90);

##Replacing bad entries with correct ones 

##Gender-F is actually female -For EXACT MATCH and changing Super + -->Super Plus

item1="\\b"+"F"+"\\b"
capchurn.Gender=capchurn.Gender.str.replace(item1,'Female')
capchurn['Gender'].value_counts().plot(kind='bar')


item1="\\b"+"M"+"\\b"
capchurn.Gender=capchurn.Gender.str.replace(item1,'Male')
capchurn['Gender'].value_counts().plot(kind='bar')


capchurn['account_segment'] = capchurn['account_segment'].str.replace('+','Plus')


capchurn['account_segment'].value_counts().plot(kind='bar')

####Encoding 

#Converting object variables to corresponding categorical codes 
for j in capchurn[['Payment','Gender','Marital_Status','Login_device']]: 
        capchurn[j] = pd.Categorical(capchurn[j]).codes 
        print('Done for Column :',j)
        
##Encoding categorical ordinal variables to numerical values as per order
encobj=OrdinalEncoder(categories=[['Regular' ,'Regular Plus' ,'Super' ,'Super Plus' ,'HNI']])
capchurn['account_segment']=encobj.fit_transform(capchurn[["account_segment"]])    
capchurn['account_segment'].value_counts().plot(kind='bar') 


####Dropping ID field 
capchurn.drop('AccountID',axis=1,inplace=True) 

  
        
####train test split 
# Copy all the predictor variables into X dataframe
all_ind_ds= capchurn.drop('Churn', axis=1)
# Copy target into the y dataframe. 
all_labels = capchurn[['Churn']]

# Split X and y into training and test set in 75:25 ratio

trainset, testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=0.30 , random_state=888)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

# trainsetnn=trainset.copy()
# train_labelsnn=train_labels.copy()
# testsetnn=testset.copy()
# test_labelsnn=test_labels.copy()        
        
 
      
#### Logistic Regression model

start_time_0 = time.monotonic()

model_logis = LogisticRegression(solver='newton-cg',max_iter=10000,verbose=True,n_jobs=2)

model_logis.fit(trainset,train_labels)

end_time_0 = time.monotonic()
print(end_time_0)
train_time_lr=timedelta(seconds=end_time_0 - start_time_0)
print("training time for NN model is ",train_time_lr)

#trainset 
pred_train=model_logis.predict(trainset) 
print(confusion_matrix(train_labels,pred_train))
log_acc_train=accuracy_score(train_labels,pred_train)
print(accuracy_score(train_labels,pred_train)) 

probs_train = model_logis.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train = probs_train[:, 1]
# calculate AUC
log_auc_train = roc_auc_score(train_labels, probs_train)
print('AUC: %.3f' % log_auc_train)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(train_labels, probs_train)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr)


##Test set
pred_test=model_logis.predict(testset)
print(confusion_matrix(test_labels,pred_test))
log_acc_test=accuracy_score(test_labels,pred_test)
print(log_acc_test)
print(accuracy_score(test_labels,pred_test))
log_recall_test=recall_score(test_labels, pred_test)
print(recall_score(test_labels, pred_test))
 
probs_test = model_logis.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
# calculate AUC
log_auc_test = roc_auc_score(test_labels, probs_test)
print('AUC: %.3f' % log_auc_test)
# calculate roc curve
test_fpr_lr, test_tpr_lr, test_thresholds_lr = roc_curve(test_labels, probs_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr_lr, test_tpr_lr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression-TestSet')



####Naive Bayes 

# ##Shapiro-wilk test for normality 
# # 𝐻0: age follows a normal distribution 
# # against 
# # 𝐻𝑎: age does not follow a normal distribution
# # if p-val<0.05 then Null hyp is rejected 

# w, p_value = stats.shapiro(election['age']) 
# print("W = {}".format(w), "p_value = {}".format(p_value))
# ##p-val too small ,we reject H0 ,age does NOT follow normality ,we need to apply trnsformation


# # BOX-COX transform training data & save lambda value
# fitted_data, fitted_lambda = stats.boxcox(election['age'])

# # creating axes to draw plots
# fig, ax = plt.subplots(1, 2)
  
# # plotting the original data(non-normal) and 
# # fitted data (normal)
# sns.distplot(election['age'], hist = False, kde = True,
#             kde_kws = {'shade': True, 'linewidth': 2}, 
#             label = "Non-Normal", color ="green", ax = ax[0])
  
# sns.distplot(fitted_data, hist = False, kde = True,
#             kde_kws = {'shade': True, 'linewidth': 2}, 
#             label = "Normal", color ="green", ax = ax[1])
  
# # adding legends to the subplots
# plt.legend(loc = "upper right")
  
# # rescaling the subplots
# fig.set_figheight(5)
# fig.set_figwidth(10)
  
# print(f"Lambda value used for Transformation: {fitted_lambda}")

# e=election.copy()
# a=fitted_data.reshape(-1,1)
# e['age']=a

# ##train and validate split with age transformed for NB (box-cox)
# all_l=e['vote']
# all_i=e.drop(["vote"],axis=1)

# # splitting data into training and test set for independent attributes
# trset,teset, tr_labels, te_labels =train_test_split(all_i, all_l, test_size=.30, random_state=400)
# print(trset.shape)
# print(tr_labels.shape)
# print(teset.shape)
# print(te_labels.shape)

start_time_2 = time.monotonic()

nb_obj=GaussianNB()
nb_model=nb_obj.fit(trainset,train_labels)

end_time_2 = time.monotonic()
print(end_time_2)
train_time_nb=timedelta(seconds=end_time_2 - start_time_2)
print("training time for NN model is ",train_time_nb)

#train
pred_train_nb=nb_model.predict(trainset)
##Confusion matrix 
print("confusion matrix for Basic NB TRAIN= \n", confusion_matrix(train_labels,pred_train_nb))
##Accuracy score 
nb_acc_train=accuracy_score(train_labels,pred_train_nb)
print(nb_acc_train)
print("Accuracy Niave Bayes model trainset =",accuracy_score(train_labels,pred_train_nb))
#AUC
probs_train = nb_model.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train = probs_train[:, 1]
# calculate AUC
nb_auc_train = roc_auc_score(train_labels, probs_train)
print('AUC: %.3f' % nb_auc_train)


#test
pred_test_nb=nb_model.predict(testset)
print("confusion matrix for Basic NB TEST=  \n", confusion_matrix(test_labels,pred_test_nb))
nb_acc_test=accuracy_score(test_labels,pred_test_nb)
print("Accuracy score of Basic/Default NB model on TEST set=",nb_acc_test)
nb_recall_test=recall_score(test_labels, pred_test_nb)
print(recall_score(test_labels, pred_test_nb))
#AUC
probs_test = nb_model.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
# calculate AUC
nb_auc_test = roc_auc_score(test_labels, probs_test)
print('AUC: %.3f' % nb_auc_test)

# calculate roc curve
test_fpr_nb, test_tpr_nb, test_thresholds_nb = roc_curve(test_labels, probs_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr_nb,test_tpr_nb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Naive Bayes-TestSet')


####Naive Bayes tuned 
####################################################################
##var_smoothing is a stability calculation to widen (or smooth) the curve and therefore account
# for more samples that are further away from the distribution mean
############################################################################
start_time1 = time.monotonic()
print(start_time1)
nb_obj=GaussianNB()

param_grid_nb = {
    'var_smoothing': np.logspace(0,-5,num=50),
            
}

nb_model2=GridSearchCV(estimator=nb_obj,param_grid=param_grid_nb,cv=10,verbose=20,scoring='accuracy')
nb_model2.fit(trainset,train_labels)

##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
train_time_nbtune=timedelta(seconds=end_time1 - start_time1)
print("training time of Naive Bayes model= ",train_time_nbtune)
print("Best paramsof Tuned Naive Bayes are =",nb_model2.best_params_)
print("Best score after gridsearch =",nb_model2.best_score_)
nb_model_bst=nb_model2.best_estimator_


pred_train_bst_nb=nb_model_bst.predict(trainset)
pred_test_bst_nb=nb_model_bst.predict(testset)
##Confusion matrix 
print("confusion matrix for TRAIN= \n", confusion_matrix(train_labels,pred_train_bst_nb))
print("confusion matrix for TEST=  \n", confusion_matrix(test_labels,pred_test_bst_nb))
##Accuracy score 
acc_score_train_bst_nb=accuracy_score(train_labels,pred_train_bst_nb)
print("Accuracy score of Tuned nb model on TRAIN set=",acc_score_train_bst_nb)
acc_score_test_bst_nb=accuracy_score(test_labels,pred_test_bst_nb)
print("Accuracy score of Tuned nb model on TEST set=",acc_score_test_bst_nb)
##AUC scores
##Train
probs_train_nb = nb_model_bst.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train_nb = probs_train_nb[:, 1]
# calculate AUC
auc_train_nb_bst = roc_auc_score(train_labels, probs_train_nb)
print('AUC on Train set for NB tuned  model: %.3f' % auc_train_nb_bst)
##test
probs_test_nb = nb_model_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test_nb = probs_test_nb[:, 1]
# calculate AUC
auc_test_nb_bst = roc_auc_score(test_labels, probs_test_nb)
print('AUC on Train set for NB tuned model: %.3f' % auc_test_nb_bst)
#recall on test
nb_recall_test_bst=recall_score(test_labels, pred_test_bst_nb)
print('Recall score on test for tuned NB model=', recall_score(test_labels, pred_test_bst_nb))

####Random Forest -Hyp tuned via Grid Search | Bagging Classifier
start_time2 = time.monotonic()
print(start_time2)

param_grid_rf = {
    'criterion': ['entropy','gini'],
    'max_depth': [3,4,6],
    'max_features': [4,10,18],
    'min_samples_leaf': [100],
    'min_samples_split': [150,50],
    'n_estimators': [50,100]
}

rfc_obj2=RandomForestClassifier(random_state=400)
rfc_model2=GridSearchCV(estimator=rfc_obj2,param_grid=param_grid_rf,cv=3,scoring='recall',verbose=30)
rfc_model2.fit(trainset,train_labels)

##Capturing training time
end_time2 = time.monotonic()
print(end_time2)
train_time_rf=timedelta(seconds=end_time2 - start_time2)
print("training time for RF model is ",train_time_rf)

print("Best params of Tuned rf Classifier are =",rfc_model2.best_params_)
print("Best score after gridsearch =",rfc_model2.best_score_)
rfc_model_bst=rfc_model2.best_estimator_

#train
pred_train_bst_rfc=rfc_model_bst.predict(trainset)
##Confusion matrix 
print("confusion matrix for TRAIN= \n", confusion_matrix(train_labels,pred_train_bst_rfc))
##Accuracy score 
acc_score_train_bst_rfc=accuracy_score(train_labels,pred_train_bst_rfc)
print("Accuracy score of Tuned rfc model on TRAIN set=",acc_score_train_bst_rfc)
#auc train
probs_train_rf = rfc_model_bst.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train_rf = probs_train_rf[:, 1]
# calculate AUC
auc_train_rf = roc_auc_score(train_labels, probs_train_rf)
print('AUC on Train set for rf Classifier tuned model: %.3f' % auc_train_rf)


##test
pred_test_bst_rfc=rfc_model_bst.predict(testset)
print("confusion matrix for TEST=  \n", confusion_matrix(test_labels,pred_test_bst_rfc))
acc_score_test_bst_rfc=accuracy_score(test_labels,pred_test_bst_rfc)
print("Accuracy score of Tuned rfc model on TEST set=",acc_score_test_bst_rfc)

#auc test
probs_test_rf = rfc_model_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test_rf = probs_test_rf[:, 1]
# calculate AUC
auc_test_rf = roc_auc_score(test_labels, probs_test_rf)
print('AUC on Train set for rf Classifier tuned model: %.3f' % auc_test_rf)

#recall on test
recall_score_test_bst_rfc=recall_score(test_labels,pred_test_bst_rfc)
print("Recall score of Tuned rfc model on TEST set=",recall_score(test_labels,pred_test_bst_rfc))


# calculate roc curve 
#train
train_fpr_rf, train_tpr_rf, train_thresholds_rf = roc_curve(train_labels, probs_train_rf)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr_rf, train_tpr_rf)
#test
test_fpr_rf, test_tpr_rf, test_thresholds_rf = roc_curve(test_labels, probs_test_rf)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr_rf, test_tpr_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Random Forest-TestSet')

#Feature impotance-Random Forest 

x55=pd.DataFrame(rfc_model_bst.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x55[0],x55.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in % For RandomForest model')
plt.title('Feature Importance Plot')
plt.show()



####XGBoost 
##Xtreme Gradient Boosting -another Gradient boosting classifier but with a host of other tunable options and parallel processing
start_time2 = time.monotonic()
print(start_time2)

param_grid_xgb = {
    'n_estimators': [500,1000],
    'max_depth': [4,6,8],
    'eta':[0.01,0.2],
    'early_stopping_rounds':[20],
    'subsample':[0.5,0.95],
    'colsubsample_bytree':[0.5,0.95]
    #'Lambda':[0.5,0.25],
    #'alpha':[0.2]
}

xgb_obj2=XGBClassifier(random_state=400,   objective='binary:logistic')
xgb_model2=GridSearchCV(estimator=xgb_obj2,param_grid=param_grid_xgb,cv=3,scoring='recall',verbose=30)
xgb_model2.fit(trainset,train_labels)

##Capturing training time
end_time2 = time.monotonic()
print(end_time2)
train_time_xgb=timedelta(seconds=end_time2 - start_time2)
print("training time for xgb model is ",train_time_xgb)

print("Best params of Tuned xgb Classifier are =",xgb_model2.best_params_)
print("Best score after gridsearch =",xgb_model2.best_score_)
xgb_model_bst=xgb_model2.best_estimator_

#train
pred_train_bst_xgb=xgb_model_bst.predict(trainset)
##Confusion matrix 
print("confusion matrix for TRAIN= \n", confusion_matrix(train_labels,pred_train_bst_xgb))
##Accuracy score 
acc_score_train_bst_xgb=accuracy_score(train_labels,pred_train_bst_xgb)
print("Accuracy score of Tuned xgb model on TRAIN set=",acc_score_train_bst_xgb)

##AUC scores
##Train
probs_train_xgb = xgb_model_bst.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train_xgb = probs_train_xgb[:, 1]
# calculate AUC
auc_train_xgb = roc_auc_score(train_labels, probs_train_xgb)
print('AUC on Train set for xgbging Classifier basic model: %.3f' % auc_train_xgb)

##test
pred_test_bst_xgb=xgb_model_bst.predict(testset)
print("confusion matrix for TEST=  \n", confusion_matrix(test_labels,pred_test_bst_xgb))
acc_score_test_bst_xgb=accuracy_score(test_labels,pred_test_bst_xgb)
print("Accuracy score of Tuned xgb model on TEST set=",acc_score_test_bst_xgb)
recall_score_test_bst_xgb=recall_score(test_labels,pred_test_bst_xgb)
print("Recall score of Tuned xgb model on TEST set=",recall_score_test_bst_xgb)
#auc
probs_test_xgb = xgb_model_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test_xgb = probs_test_xgb[:, 1]
# calculate AUC
auc_test_xgb = roc_auc_score(test_labels, probs_test_xgb)
print('AUC on TEST set for xgbging Classifier  basic model: %.3f' % auc_test_xgb)

# calculate roc curve 
#train
train_fpr_xgb, train_tpr_xgb, train_thresholds_xgb = roc_curve(train_labels, probs_train_xgb)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr_xgb, train_tpr_xgb)
#test
test_fpr_xgb, test_tpr_xgb, test_thresholds_xgb = roc_curve(test_labels, probs_test_xgb)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr_xgb, test_tpr_xgb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve XGBoost-TestSet')


#Feature Importance 

plot_importance(xgb_model_bst)

x66=pd.DataFrame(xgb_model_bst.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x66[0],x66.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in % For XGBoost model')
plt.title('Feature Importance Plot')
plt.show()



#### NeuralNet -Multilayer perceptron 

sobj=StandardScaler()
data_scaled=sobj.fit_transform(all_ind_ds)
data_scaled_df=pd.DataFrame(data=data_scaled,columns=all_ind_ds.columns)
print(data_scaled_df.head(5))
frames2=[data_scaled_df,all_labels]

trainsetnn,testsetnn, train_labelsnn, test_labelsnn =train_test_split(data_scaled_df, all_labels, test_size=.30, random_state=444)
print(trainsetnn.shape)
print(train_labelsnn.shape)
print(testsetnn.shape)
print(test_labelsnn.shape)



####SMOTE -NN
print(train_labelsnn.value_counts())

print(test_labelsnn.value_counts())

print(all_labels.value_counts())
##Overall 17 % Positive class-1 and 83% Negative class -0
oversample=SMOTE()

trainsetnn,train_labelsnn=oversample.fit_resample(trainsetnn,train_labelsnn)

start_time3 = time.monotonic()
print(start_time3)

nn_cl=MLPClassifier(random_state=444)
param_grid3 = {
    'hidden_layer_sizes': [512,1024],
    'max_iter': [500,1000],
    'solver': ['adam'],
    'tol': [0.001]
}

nn_mod1 = GridSearchCV(estimator = nn_cl, param_grid = param_grid3, cv = 10,verbose=30,scoring='recall')

nn_mod1.fit(trainsetnn,train_labelsnn)

###capturing model training time 
##Capturing training time
end_time3 = time.monotonic()
print(end_time3)
train_time_nn=timedelta(seconds=end_time3 - start_time3)
print("training time for NN model is ",train_time_nn)

print("The best params of the NN tuned models are =",nn_mod1.best_params_)
nn_mod1_bst=nn_mod1.best_estimator_

#train
pred_train_nn=nn_mod1_bst.predict(trainsetnn)
print("Confusion matrix NN model on train set ",confusion_matrix(train_labelsnn,pred_train_nn))
nn_acc_train=accuracy_score(train_labelsnn,pred_train_nn)
print(accuracy_score(train_labelsnn,pred_train_nn))
# AUC and ROC for the train data
# predict probabilities
probs_nn = nn_mod1_bst.predict_proba(trainsetnn)
# keep probabilities for the positive outcome only
probs_nn = probs_nn[:, 1]
# calculate AUC
auc_nn_train = roc_auc_score(train_labelsnn, probs_nn)
print('AUC: %.3f' % auc_nn_train)
# calculate roc curve
fpr3, tpr3, thresholds = roc_curve(train_labelsnn, probs_nn)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.')
# show the plot
plt.show()



#Test
pred_test_nn=nn_mod1_bst.predict(testsetnn)




####recreating testset with predictions 
pred_unk=pd.Series(pred_test_nn.astype(int),name="Churn")
pred_unk=pd.DataFrame(pred_unk)


retest=pd.concat([testset.reset_index(drop=True),pred_unk],axis=1)

retest.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\capstone project\predicted.csv',index=False)

print(confusion_matrix(test_labelsnn,pred_test_nn))
print(classification_report(test_labelsnn,pred_test_nn))

nn_acc_test=accuracy_score(test_labelsnn,pred_test_nn)
nn_recall_test=recall_score(test_labelsnn,pred_test_nn)                         
print("Accuracy Score on Test set for NN Model =",accuracy_score(test_labelsnn,pred_test_nn))
print("Recall Score on Test set for NN Model =",recall_score(test_labelsnn,pred_test_nn))
# AUC and ROC for the test data
# predict probabilities
probs_nn_test = nn_mod1_bst.predict_proba(testsetnn)
# keep probabilities for the positive outcome only
probs_nn_test = probs_nn_test[:, 1]
# calculate AUC
auc_nn_test = roc_auc_score(test_labelsnn, probs_nn_test)
print('AUC: on the TEST SET for NN model= %.3f' % auc_nn_test)
# calculate roc curve
fpr3_test, tpr3_test, thresholds_test = roc_curve(test_labelsnn, probs_nn_test)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3_test, tpr3_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve NeuralNet-TestSet')
# show the plot
plt.show()









print("Rounder value =",round(auc_nn_test)*100)

####result collation 
index=['Train Accuracy','Train AUC', 'Test AUC','Test Accuracy','Test Recall','Training Time']
result = pd.DataFrame({
        'Logistic Regression':[log_acc_train,log_auc_train,log_auc_test,log_acc_test,log_recall_test,  train_time_lr],
         'Naive Bayes':[nb_acc_train,nb_auc_train,nb_auc_test,nb_acc_test,nb_recall_test, train_time_nb],
         'Naive Bayes Tuned':[acc_score_train_bst_nb,  auc_train_nb_bst,  auc_test_nb_bst,  acc_score_test_bst_nb,  nb_recall_test_bst, train_time_nbtune],
        'Random Forest':[acc_score_train_bst_rfc,  auc_train_rf,   auc_test_rf,   acc_score_test_bst_rfc,  recall_score_test_bst_rfc,  train_time_rf],
        'XGBoost':[acc_score_train_bst_xgb,   auc_train_xgb,  auc_test_xgb,   acc_score_test_bst_xgb,    recall_score_test_bst_xgb,  train_time_xgb],
        'Neural Net':[nn_acc_train,   auc_nn_train,   auc_nn_test,    nn_acc_test,    nn_recall_test,   train_time_nn]
        
        },index=index)
print(round(result,5))

####Combined ROC curve on TEST SET 
##Plotting ROC curve for comparison for Test set ONLY 
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(test_fpr_lr, test_tpr_lr,color='pink',label="Logistic")
plt.plot(test_fpr_nb, test_tpr_nb,color='blue',label="NaiveBayes")
plt.plot(test_fpr_rf, test_tpr_rf,color='yellow',label="RF")
plt.plot(test_fpr_xgb, test_tpr_xgb,color='green',label="XGB")
plt.plot(test_fpr_xgb, test_tpr_xgb,color='red',label="ANN")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')

####collated result file generation
snapshotdate = datetime.datetime.today().strftime('%d-%m-%Y_%H_%M_%S')
print(snapshotdate)
myfile_name='ResultFile_'+snapshotdate+'.csv'
print(myfile_name)
p='E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\capstone project\\result file'
print("printing output file for collated result file ")
result.to_csv(p+myfile_name,index=False)


