# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:37:34 2021

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

##Basic EDA ::Columns dtypes,shape,unique values ,null or missing values ,outliers,any unique id in any column,
##correlated columns,same value or Zero variance columns;; DOM -SMET 

usheart_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week 12-RandomForest\\US_Heart_Patients.csv')

usheart=usheart_orig.copy()

usheart['Heart-Att'].value_counts()
##Imbalanced data set 
    
print(usheart.dtypes )
print(usheart.shape)       
 ##Duplicates 
dups=usheart.duplicated()     
print("No of duplicates in this dataset=",dups.sum())
usheart[dups]     
        
##Outliers
usheart.boxplot()
plt.xticks(rotation=90);
##outliers exist but CART and RF not sensitive to outliers

##Missing values 
usheart.isnull().sum()
#imputing continuous columns only with median 
for i in usheart[['age', 'cigsPerDay', 'tot cholesterol', 'Systolic BP', 'Diastolic BP','BMI','heartRate','glucose']]:
    median = usheart[i].median()
    usheart[i].replace(np.nan, median, inplace= True)
    
    usheart.isnull().sum()
    
##Dropping rows where categorical column is missing 
usheart.dropna(inplace=True)
print(usheart.shape)
# Decision tree in Python can take only numerical / categorical colums. It cannot take string / object types. 
# The following code loops through each column and checks if the column type is object then converts those columns
# into categorical with each distinct value becoming a category or code.

##Scaling -->doesnot impact CART and RF
##Multicollinearity 
plt.figure(figsize=(12,7))
sns.heatmap(usheart.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()

for j in usheart.columns: 
    if usheart[j].dtype == 'object':
        usheart[j] = pd.Categorical(usheart[j]).codes 

##########K-Means data prep#########################
clust_then_pred=usheart

##Outlier treatment -reqd for clustering 
clust_then_pred.boxplot()
plt.xticks(rotation=90);

for feature in clust_then_pred[['age', 'cigsPerDay', 'tot cholesterol', 'Systolic BP', 'Diastolic BP','BMI','heartRate','glucose']]: 
    lr,ur=remove_outlier(clust_then_pred[feature])
    clust_then_pred[feature]=np.where(clust_then_pred[feature]>ur,ur,clust_then_pred[feature])
    clust_then_pred[feature]=np.where(clust_then_pred[feature]<lr,lr,clust_then_pred[feature])

clust_then_pred.boxplot()
plt.xticks(rotation=90);

# ##Scaling required for clustering ::using zscore scaling 
ctp_all_ind_ds=clust_then_pred.drop(["Heart-Att"],axis=1)

sobj=StandardScaler()
data_scaled=sobj.fit_transform(ctp_all_ind_ds)
data_scaled=pd.DataFrame(data_scaled,columns=ctp_all_ind_ds.columns)
data_scaled.boxplot()
plt.xticks(rotation = 90);
# k_means2 = KMeans(n_clusters = 2,random_state=1)
# k_means2.fit(data_scaled)
# k_means2.labels_
wss =[] 
for i in range(1,15):
     KM = KMeans(n_clusters=i,random_state=0)
     KM.fit(data_scaled)
     wss.append(KM.inertia_)
 
plt.plot(range(1,15), wss)
plt.grid()
plt.show()
k_means6 = KMeans(n_clusters = 6,random_state=0)
k_means6.fit(data_scaled)
k_means6.inertia_
labels6=k_means6.labels_
labels6
usheart['ClusterNum']=labels6
# from sklearn.metrics import silhouette_samples, silhouette_score
# silhouette_score(data_scaled,lbls)


all_ind_ds=usheart.drop(["Heart-Att"],axis=1)
all_labels=usheart.pop("Heart-Att")
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=0)
testset.shape
##Baseline model
test_labels.value_counts()
##Baseline dumb model which predicts all NO/0 for target Hear-Att will have acc=84.8% 
##On test set baseline model will have acc=85.29% 
##Our model should at least beat pure guesswork model

########RANDOM FOREST (cluster as feature with default parameters)#####################
rfc_obj3=RandomForestClassifier(random_state=0)
rfc_mod3=rfc_obj3.fit(trainset,train_labels)
pred_test_normal3=rfc_mod3.predict(testset)
confusion_matrix(test_labels,pred_test_normal3)
print(classification_report(test_labels,pred_test_normal3))
print(accuracy_score(test_labels,pred_test_normal3))
# AUC and ROC for the test data
# predict probabilities
probs3 = rfc_mod3.predict_proba(testset)
# keep probabilities for the positive outcome only
probs3 = probs3[:, 1]
# calculate AUC
auc3 = roc_auc_score(test_labels, probs3)
print('AUC for default parameters model on test set: %.3f' % auc3)
# calculate roc curve
fpr3, tpr3, thresholds3 = roc_curve(test_labels, probs3)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.')
# show the plot
plt.show()

########RANDOM FOREST (Grid search TUNED parameters-WITH Cluster)#####################
##creating rf model with hyper parameter tuning using grid search
param_grid = {
    'criterion': ['entropy','gini'],
    'max_depth': [7, 10],
    'max_features': [0,4,6,2,8,0.2],
    'min_samples_leaf': [50, 100],
    'min_samples_split': [150, 300],
    'n_estimators': [10,15,25,50]
}

rfc_obj4=RandomForestClassifier(random_state=0)
rfc_mod_gs4=GridSearchCV(estimator=rfc_obj4,param_grid=param_grid,cv=3,scoring=make_scorer(recall_score),verbose=10)
rfc_mod_gs4.fit(trainset,train_labels)
rfc_mod_gs4.best_params_
rfc_mod_bst4=rfc_mod_gs4.best_estimator_
pred_test_gs4=rfc_mod_bst4.predict(testset)
confusion_matrix(test_labels,pred_test_gs4)
print(accuracy_score(test_labels,pred_test_gs4))
print(classification_report(test_labels,pred_test_gs4))
# AUC and ROC for the test data
# predict probabilities
probs4 = rfc_mod_bst4.predict_proba(testset)
# keep probabilities for the positive outcome only
probs4 = probs4[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc4 = roc_auc_score(test_labels, probs4)
print('AUC for grid searched model with cluster as feature : %.3f' % auc4)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr4, tpr4, thresholds4 = roc_curve(test_labels, probs4)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr4, tpr4, marker='.')
# show the plot
plt.show()


##CART (entropy) WITH Cluster as feature
cart_mod5 = DecisionTreeClassifier(random_state=0,max_depth=7,criterion="entropy")      
cart_mod5.fit(trainset,train_labels)
pred_cart_test5 = cart_mod5.predict(testset)
#pred_cart_train5=cart_mod5.predict(trainset)       

##Feature importance 
x5=pd.DataFrame(cart_mod5.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x5[0],x5.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()
##Accuracy 
print('Accuracy Score is',round(accuracy_score(test_labels, pred_cart_test5),2)*100,'%')
##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(test_labels,cart_mod5.predict_proba(testset)[:,1]),2)*100,'%')




##Without clustering models 
trainset=trainset.drop(["ClusterNum"],axis=1)
testset=testset.drop(["ClusterNum"],axis=1)


########RANDOM FOREST (Default parameters-Without cluster)#####################

##Creating rf model with default paramters (IMPUTED with Median)
rfc_obj1=RandomForestClassifier(random_state=0)
rfc_mod1=rfc_obj1.fit(trainset,train_labels)
pred_test_normal=rfc_mod1.predict(testset)
confusion_matrix(test_labels,pred_test_normal)
#print(classification_report(test_labels,pred_test_normal))
print(accuracy_score(test_labels,pred_test_normal))
# AUC and ROC for the test data
# predict probabilities
probs1 = rfc_mod1.predict_proba(testset)
# keep probabilities for the positive outcome only
probs1 = probs1[:, 1]
# calculate AUC
auc1 = roc_auc_score(test_labels, probs1)
print('AUC for default parameters model on test set: %.3f' % auc1)
# calculate roc curve
fpr1, tpr1, thresholds1 = roc_curve(test_labels, probs1)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker='.')
# show the plot
plt.show()

########RANDOM FOREST (Grid search TUNED parameters-Without Cluster)#####################
##creating rf model with hyper parameter tuning using grid search
param_grid = {
    'criterion': ['entropy','gini'],
    'max_depth': [7, 10],
    'max_features': [0,4,6,2,8,0.2],
    'min_samples_leaf': [50, 100],
    'min_samples_split': [150, 300],
    'n_estimators': [10,15,25,50]
}

rfc_obj2=RandomForestClassifier(random_state=0)
rfc_mod_gs2=GridSearchCV(estimator=rfc_obj2,param_grid=param_grid,cv=3,scoring=make_scorer(recall_score),verbose=2)
rfc_mod_gs2.fit(trainset,train_labels)
rfc_mod_gs2.best_params_
rfc_mod_bst2=rfc_mod_gs2.best_estimator_
pred_test_gs2=rfc_mod_bst2.predict(testset)
confusion_matrix(test_labels,pred_test_gs2)
print(accuracy_score(test_labels,pred_test_gs2))
#print(classification_report(test_labels,pred_test_normal))
# AUC and ROC for the test data
# predict probabilities
probs2 = rfc_mod_bst2.predict_proba(testset)
# keep probabilities for the positive outcome only
probs2 = probs2[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc2 = roc_auc_score(test_labels, probs2)
print('AUC for grid searched model: %.3f' % auc2)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr2, tpr2, thresholds2 = roc_curve(test_labels, probs2)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr2, tpr2, marker='.')
# show the plot
plt.show()


########CART (default parameters)#####################

cart_mod1 = DecisionTreeClassifier(random_state=0,max_depth=7,criterion="gini")      
cart_mod1.fit(trainset,train_labels)
pred_cart_test = cart_mod1.predict(testset)
pred_cart_train=cart_mod1.predict(trainset)       

##Feature importance 
x=pd.DataFrame(cart_mod1.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x[0],x.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()
##Accuracy 
print('Accuracy Score is',round(accuracy_score(test_labels, pred_cart_test),2)*100,'%')
##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(test_labels,cart_mod1.predict_proba(testset)[:,1]),2)*100,'%')
print('Area Under the Curve is',roc_auc_score(test_labels,cart_mod1.predict_proba(testset)[:,1]),2)*100

################Cart with Entropy##########################
cart_mod2 = DecisionTreeClassifier(random_state=0,max_depth=7,criterion="entropy")      
cart_mod2.fit(trainset,train_labels)
pred_cart_test2 = cart_mod2.predict(testset)
pred_cart_train2=cart_mod2.predict(trainset)       

##Feature importance 
x2=pd.DataFrame(cart_mod2.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x2[0],x2.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()
##Accuracy 
print('Accuracy Score is',round(accuracy_score(test_labels, pred_cart_test2),2)*100,'%')
##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(test_labels,cart_mod2.predict_proba(testset)[:,1]),2)*100,'%')
















