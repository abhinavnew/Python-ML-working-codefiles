# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:42:57 2021

@author: Abhinav.Bajpai
"""


import gc
##Clear variable/objects from workspace to free up memory
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
dir()
import numpy as np 
print("numpy path is",np.__path__)
print('The numpy version is {}.'.format(np.__version__))

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
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The matplotlib version is {}.'.format(mp.__version__))
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics ##for rmse
import statsmodels.formula.api as smf ##LinReg using statsmodel
from sklearn.linear_model import LogisticRegression

def getupperlower_outlier(col):
    sorted(col)
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
def drop_constant_columns_4(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    keep_columns = dataframe.columns[dataframe.nunique()>1]
    a=dataframe.loc[:,keep_columns].copy()
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
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)
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

lie_orig =pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 15-Logistic\liedetector.csv')
lie=lie_orig.copy()

print(lie.shape)
print(lie.head(5))
print(lie.dtypes)
print(lie.describe())
print(lie.info())

##Univariate analysis 
print(lie['X2'].nunique())
print(lie['X3'].nunique())
print(lie['X4'].nunique())

#Histograms for continuous variables
sns.distplot(lie.X2,bins=10)
sns.distplot(lie.X3,bins=10)
sns.distplot(lie.X4,bins=10)
sns.distplot(lie.X5,bins=10)

##Categorical Target
lie['X6'].value_counts().plot(kind='bar')

lie.drop(["X1"],axis=1,inplace=True)

##Duplicates
dups=lie.duplicated()
print("The num of duplicate records are =",dups.sum())
lie[dups]
lie.drop_duplicates(inplace=True)
print(lie.shape)

##Missing values
lie.isnull().sum()
lie.info()
##bad values
bad_values = ['\?','NA','\@','None','NaN','Nan','nan','Missing','-99','-999','\??','\???'] 
cat_col,num_col=get_num_cat_colnames(lie)
print(cat_col)
print(num_col)

for feature in lie.columns: 
    if lie[feature].dtype == 'float64': 
        print(feature)
        print(lie[feature].value_counts())
        print('\n')
##Outliers
lie.boxplot()
plt.xticks(rotation=90);
ll1,ul1=getupperlower_outlier(lie['X4'])
out_X4=lie.loc[(lie['X4']>ul1)|(lie['X4']<ll1),]
print(out_X4)
#treatment winsorization 
lie['X4']=treat_outlier_ul_ll_winsor(lie['X4'])
lie['X5']=treat_outlier_ul_ll_winsor(lie['X5'])
lie.boxplot()
plt.xticks(rotation=90);


##Multi
plt.figure(figsize=(12,7))
sns.heatmap(lie.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()


##Pairplot 
sns.pairplot(lie,hue='X6',diag_kind='kde')
plt.show()

##Baseline 
lie['X6'].value_counts()
##Baseline model accuracy ~58.9%

##Splitting into train ,test 
all_labels=lie['X6']
all_ind_ds=lie.drop(["X6"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=777)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)




# Fit the Logistic Regression model
model_logis = LogisticRegression(solver='newton-cg',max_iter=10000,penalty='none',verbose=True,n_jobs=2)
model_logis.fit(all_ind_ds,all_labels)
pred_all=model_logis.predict(all_ind_ds)


confusion_matrix(all_labels,pred_all)
accuracy_score(all_labels,pred_all)
recall_score(all_labels,pred_all)
probs_train = model_logis.predict_proba(all_ind_ds)
# keep probabilities for the positive outcome only
probs_train = probs_train[:, 1]
# calculate AUC
auc_train = roc_auc_score(all_labels, probs_train)
print('AUC: %.3f' % auc_train)
# calculate roc curve
train_fpr, train_tpr, train_thresholds = roc_curve(all_labels, probs_train)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr, train_tpr)


##prediction 
pred_test = model_logis.predict(testset)
pred_train=model_logis.predict(trainset)   

##probabilities
ytest_pred_prob=model_logis.predict_proba(testset)
pd.DataFrame(ytest_pred_prob).head()



##Model evaluation 

model_logis.score(trainset,train_labels)

model_logis.score(testset,test_labels)

##AUC ROC for train data 
# predict probabilities
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

##right threshold 
optimal_idx = np.argmax(train_tpr - train_fpr)
optimal_threshold = train_thresholds[optimal_idx]
print(optimal_threshold)

##Test set prediction as per optimal thresh
y_test_pred=[]
for i in range(0,len(probs)):
    if np.array(probs)[i]>optimal_threshold:
        a=1
    else:
        a=0
    y_test_pred.append(a)






##AUC ROC on Testdata 
### predict probabilities
probs = model_logis.predict_proba(testset)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
test_auc = roc_auc_score(test_labels,probs)
print('AUC: %.3f' % test_auc)
# calculate roc curve
test_fpr, test_tpr, test_thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr)

##
confusion_matrix(test_labels,pred_test)
accuracy_score(test_labels,pred_test)
recall_score(test_labels,pred_test)
print(classification_report(test_labels,pred_test))
print(classification_report(test_labels,y_test_pred))

