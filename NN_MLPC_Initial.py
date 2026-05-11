# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:44:47 2021

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


bank_orig1=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week13-NeuralNet\\Bank Dataset.csv')
bank=bank_orig1.copy()

bank=bank.drop(['ID'],axis=1)


dups=bank.duplicated()
print(dups.sum())

bank.boxplot()
plt.xticks(rotation=90);

bank.isnull().sum()


all_ind_ds=bank.drop(["Personal_Loan"],axis=1)
all_labels=bank['Personal_Loan']
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=27)
testset.shape
##Baseline model
Baseline_acc=4504/(4504+478)
print("Accuracy of BaseLine Model is ",Baseline_acc)

##Scaling reqd for neuralnet 
sobj=StandardScaler()
data_scaled_trainset=sobj.fit_transform(trainset)
data_scaled_testset=sobj.transform(testset)

# #data_scaled_trainset.boxplot()
# #plt.xticks(rotation = 90);
nn_mod1=MLPClassifier(hidden_layer_sizes=100,max_iter=5000,solver='sgd',verbose=True,random_state=27,tol=0.01)
nn_mod1.fit(data_scaled_trainset,train_labels)
pred_test_nn=nn_mod1.predict(data_scaled_testset)
print(confusion_matrix(test_labels,pred_test_nn))
print(accuracy_score(test_labels,pred_test_nn))

# AUC and ROC for the test data
# predict probabilities
probs = nn_mod1.predict_proba(data_scaled_testset)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


###############################QUIZ
bc_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week13-NeuralNet\\ANN_Quiz_Data_Breast_Cancer.csv')
bc=bc_orig.copy()

#duplicates
dups=bc.duplicated()
print(dups.sum())

#Outliers
bc.boxplot()
plt.xticks(rotation=90);

##Missing values
bc.isnull().sum()

##Multicollinearity
plt.figure(figsize=(12,7))
sns.heatmap(bc.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()

corrmat=bc.corr()
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
    rotation=45,
    horizontalalignment='right'
);

# corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat_homeloan.csv")

##Train/Test divide 

all_ind_ds=bc.drop(["Target"],axis=1)
all_labels=bc['Target']
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.20, random_state=0)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

##Baseline model
Baseline_acc=357/(357+212)
print("Accuracy of BaseLine Model is ",Baseline_acc)

##Scaling reqd for neuralnet 
sobj=StandardScaler()
data_scaled_trainset=sobj.fit_transform(trainset)
data_scaled_testset=sobj.transform(testset)

# #data_scaled_trainset.boxplot()
# #plt.xticks(rotation = 90);
nn_mod2=MLPClassifier(hidden_layer_sizes=500,verbose=True,random_state=0,tol=0.0001)
nn_mod2.fit(trainset,train_labels)
pred_test_nn1=nn_mod2.predict(trainset)
pred_test_nn2=nn_mod2.predict(testset)
print(accuracy_score(test_labels,pred_test_nn2))
print(confusion_matrix(test_labels,pred_test_nn2))
print(recall_score(test_labels,pred_test_nn2))

print(accuracy_score(train_labels,pred_test_nn1))

# AUC and ROC for the test data
# predict probabilities
probs = nn_mod2.predict_proba(data_scaled_testset)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(test_labels, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
