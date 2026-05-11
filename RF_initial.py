# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:52:28 2021

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
from sklearn.tree import DecisionTreeClassifier   ##DT for CART
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
 ##for clustering
from sklearn.cluster import KMeans  ## for k means
sns.set(color_codes=True) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

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


def missvalue_replacenull(dframe,cat_colnames):
    for i in cat_colnames :
        a=dframe[dframe[i].str.contains('\?') | dframe[i].str.contains("NA") | dframe[i].str.contains("NaN")]
        print("records containing ? NA NAN etc are  ",a.shape )
        print("For column name =",i)
        idx=a.index
        

    
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
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)

##Basic EDA ::Columns dtypes,shape,unique values ,null or missing values ,outliers,any unique id in any column,
##correlated columns,same value or Zero variance columns;; DOM -SMET 

bank_orig=pd.read_csv(r'C:\Abhinav B\Kaggle\Python related\DSBA related\Week 12-RandomForest\Banking Dataset.csv')

bank=bank_orig.copy()

bank['Target'].value_counts()
##Imbalanced data set 

# Decision tree in Python can take only numerical / categorical colums. It cannot take string / object types. 
# The following code loops through each column and checks if the column type is object then converts those columns
# into categorical with each distinct value becoming a category or code.


for j in bank.columns: 
    if bank[j].dtype == 'object':
        bank[j] = pd.Categorical(bank[j]).codes 
        
 ##Duplicates 
       
        
        


all_ind_ds=bank.drop(["Target","Cust_ID"],axis=1)

all_labels=bank.pop("Target")

# splitting data into training and test set for independent attributes


trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=0)

##Creating rf model with default paramters
rfc_obj=RandomForestClassifier(random_state=0)
rfc_mod1=rfc_obj.fit(trainset,train_labels)

pred_test_normal=rfc_mod1.predict(testset)

confusion_matrix(test_labels,pred_test_normal)
acc_normal=(5457+65)/(5457+65+28+450)
print(acc_normal)
print(classification_report(test_labels,pred_test_normal))

# AUC and ROC for the test data


# predict probabilities
probs1 = rfc_mod1.predict_proba(testset)
# keep probabilities for the positive outcome only
probs1 = probs1[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc1 = roc_auc_score(test_labels, probs1)
print('AUC for default parameters model on test set: %.3f' % auc1)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr1, tpr1, thresholds1 = roc_curve(test_labels, probs1)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker='.')
# show the plot
plt.show()






##creating rf model with hyper parameter tuning using grid search
param_grid = {
    'max_depth': [7, 10],
    'max_features': [4, 6],
    'min_samples_leaf': [50, 100],
    'min_samples_split': [150, 300],
    'n_estimators': [301, 501]
}

rfc_obj2=RandomForestClassifier()
rfc_mod_gs=GridSearchCV(estimator=rfc_obj2,param_grid=param_grid,cv=3)

rfc_mod_gs.fit(trainset,train_labels)
rfc_mod_gs.best_params_
rfc_mod_bst=rfc_mod_gs.best_estimator_

pred_test_gs=rfc_mod_bst.predict(testset)

confusion_matrix(test_labels,pred_test_gs)
acc_gs=(5474+27)/(5474+27+11+488)
print(acc_gs)
print(classification_report(test_labels,pred_test_normal))

# AUC and ROC for the test data


# predict probabilities
probs = rfc_mod_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_labels, probs)
print('AUC for grid searched model: %.3f' % auc)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

