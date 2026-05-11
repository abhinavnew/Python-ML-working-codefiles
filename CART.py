# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:37:40 2021

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

bcw_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week11-CART_ROC\\breast-cancer-wisconsin.csv')
bcw=bcw_orig.copy()
##duplicates
dups=bcw.duplicated()
print("no. of duplicated records are ",dups.sum())
bcw.drop_duplicates(inplace=True)

##Outliers
bcw.boxplot()
plt.xticks(rotation=90)

##Missing values and bad values 
bcw.isnull().sum()

catcols,numcols=get_num_cat_colnames(bcw)
bcw[bcw['Bare Nuclei']!='?']['Bare Nuclei'].astype('int').median()
bcw.replace('?',1,inplace=True)


bcw['Bare Nuclei']=bcw['Bare Nuclei'].astype('int64')


##scaling not required as CART is not impacted by diff scales 

##Multicollinearity 

corrmat=bcw.corr()
corrmat[np.abs(corrmat)<.40] = 0
##masking upper triangle as its distracting   sns.diverging_palette(20, 220, n=200)
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
##Using diverging colors and rotating x axis labels
ax2=sns.heatmap(corrmat, 
            annot=False,
            vmin=-1,vmax=1,center=0,
            cmap='coolwarm',
            square=True,mask=mask)

ax2.set_xticklabels(
    ax2.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);

corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat.csv")

plt.figure(figsize=(12,7))
sns.heatmap(bcw.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()



##train and Test set 

X = bcw.drop(['Class','Sample code number  '],axis=1)
y = bcw['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


dtree = DecisionTreeClassifier(random_state=123)

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

##Feature importance 
x=pd.DataFrame(dtree.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()

##Confusion matrix 
sns.heatmap(confusion_matrix(y_test,predictions),annot=True, fmt='d', cbar=False,cmap='YlGnBu')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()


##Accuracy 
print('Accuracy Score is',round(accuracy_score(y_test, predictions),2)*100,'%')


##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(y_test,dtree.predict_proba(X_test)[:,1]),2)*100,'%')

##ROC 
dt_fpr, dt_tpr,_=roc_curve(y_test,dtree.predict_proba(X_test)[:,1])

plt.plot(dt_fpr,dt_tpr, marker='o', label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

# Uniformity of cell size plays an important part in deciding whether the Tumor is benign or malignant (Highest feature importance)

# Cases where the Tumor is Actually Malignant , there are 9 instances where model predicted the Tumor to be Benign

# Cases where the Tumor is Actually Benign but model predicted them to be Malignant are 6

# The Area Under the Curve is 92%

# Accuracy score on Testing set is 93%


##Quiz

heart_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week11-CART_ROC\\heart.csv')

heart=heart_orig.copy()

##Duplicates 
heart.shape
heart.dtypes
heart.head(10)
heart['target'].value_counts()
##Check for imbalance 

dups=heart.duplicated()
print("no of duplicates are ",dups.sum())
heart[dups]
heart.drop_duplicates(inplace=True)


##Outliers
heart.boxplot()
plt.xticks(rotation=90)

##Missing values
heart.isnull().sum()
catc,numc=get_num_cat_colnames(heart)

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

##scaling not required for CART

##Multicollinearity 
plt.figure(figsize=(12,7))
sns.heatmap(heart.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()


corrmat=heart.corr()
corrmat[np.abs(corrmat)<.40] = 0
##masking upper triangle as its distracting   sns.diverging_palette(20, 220, n=200)
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
##Using diverging colors and rotating x axis labels
ax2=sns.heatmap(corrmat, 
            annot=False,
            vmin=-1,vmax=1,center=0,
            cmap='coolwarm',
            square=True,mask=mask)

ax2.set_xticklabels(
    ax2.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
);

##

##train and Test set 

X = heart.drop(['target'],axis=1)
y = heart['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


dtree = DecisionTreeClassifier(random_state=0,max_depth=7,criterion="gini")

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
pred_train=dtree.predict(X_train)

##Feature importance 
x=pd.DataFrame(dtree.feature_importances_*100,index=X_train.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()

##Confusion matrix 
sns.heatmap(confusion_matrix(y_test,predictions),annot=True, fmt='d', cbar=False,cmap='YlGnBu')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()


sns.heatmap(confusion_matrix(y_train,pred_train),annot=True, fmt='d', cbar=False,cmap='YlGnBu')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()



##Accuracy 
print('Accuracy Score is',round(accuracy_score(y_test, predictions),2)*100,'%')

print('Accuracy Score is',round(accuracy_score(y_train, pred_train),2)*100,'%')
##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(y_test,dtree.predict_proba(X_test)[:,1]),2)*100,'%')
print('Area Under the Curve is',roc_auc_score(y_test,dtree.predict_proba(X_test)[:,1]),2)*100

##ROC 
dt_fpr, dt_tpr,_=roc_curve(y_test,dtree.predict_proba(X_test)[:,1])

plt.plot(dt_fpr,dt_tpr, marker='o', label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()