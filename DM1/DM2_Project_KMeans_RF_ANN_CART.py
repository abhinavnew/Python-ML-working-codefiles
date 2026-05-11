# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:15:52 2021

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
from sklearn.neural_network import MLPClassifier


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

insurance_orig =pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\Kmeans_CART_RF_ANN_project\insurance_part2_data.csv')
insurance=insurance_orig.copy()

# # An Insurance firm providing tour insurance is facing higher claim frequency. 
# The management decides to collect data from the past few years.
#  You are assigned the task to make a model which predicts the claim status 
#  and provide recommendations to management. Use CART, RF & ANN and 
#  compare the models' performances in train and test sets.

# # 2.1 Read the data, do the necessary initial steps, and exploratory data analysis 
# (Univariate, Bi-variate, and multivariate analysis).

print(insurance.shape)
print(insurance.head(5))
print(insurance.dtypes)
print(insurance.describe())
print(insurance.info())

##Univariate analysis 
print(insurance['Type'].unique())
print(insurance['Channel'].unique())
print(insurance['Product Name'].unique())
print(insurance['Agency_Code'].unique())
print(insurance['Destination'].unique())

##Agency_code ,Type,Channel,Product Name,Destination are categorical variables 
##Age,commision,Duration ,Sales are continuous variables
print(insurance.dtypes)
print(insurance.info())
print(insurance[['Age','Duration','Sales','Commision']].describe())

##remove Duplciates

dups=insurance.duplicated()
print("The num of duplicate records are =",dups.sum())
insurance[dups]
insurance.drop_duplicates(inplace=True)
print(insurance.shape)

##Missing values
insurance.isnull().sum()
insurance.info()
#get any bad values
bad_values = ['\?','NA','\@','None','NaN','Nan','nan','Missing','-99','-999','\??','\???'] 
cat_col,num_col=get_num_cat_colnames(insurance)
##c=select_badvalues_rows(insurance,cat_col,bad_values)
##print(c)

##Outliers
insurance.boxplot()
plt.xticks(rotation=90);

ll1,ul1=getupperlower_outlier(insurance['Duration'])
out_duration=insurance.loc[(insurance['Duration']>ul1)|(insurance['Duration']<ll1),]
ll2,ul2=getupperlower_outlier(insurance['Sales'])
out_sales=insurance.loc[(insurance['Sales']>ul2)|(insurance['Sales']<ll2),]
ll3,ul3=getupperlower_outlier(insurance['Commision'])
out_commision=insurance.loc[(insurance['Commision']>ul3)|(insurance['Commision']<ll3),]
ll4,ul4=getupperlower_outlier(insurance['Age'])
out_age=insurance.loc[(insurance['Age']>ul4)|(insurance['Age']<ll4),]
print(out_duration)
print(out_sales)
print(out_commision)
print(out_age)
##Bad values 
insurance.loc[(insurance['Duration']<0),]
insurance.loc[(insurance['Age']<0),]
insurance.loc[(insurance['Commision']<0),]
insurance.loc[(insurance['Sales']<0),]
##replace negative values with median
insurance['Duration']=np.where(insurance['Duration']<0,insurance['Duration'].median(),insurance['Duration'])
##Outlier treatement is not required for CART /RF tree based models but required for NN based models

#Histograms for continuous variables
sns.distplot(insurance.Duration,bins=10)
sns.distplot(insurance.Sales,bins=10)
sns.distplot(insurance.Commision,bins=10)
sns.distplot(insurance.Age,bins=10)
##stats
insurance['Duration'].describe()
##Checking skewness for symmetry 
print(insurance['Duration'].skew())
print(insurance['Sales'].skew())
print(insurance['Commision'].skew())
print(insurance['Age'].skew())

##Duration is most highly skewed ,followed by Commision and Sales ;Age is more or less symmetric as shown by values of skewness


##Categorical variables
#Count bar plots for categorical variables 
insurance['Agency_Code'].value_counts().plot(kind='bar')
sns.countplot(insurance_orig['Agency_Code'],hue=insurance_orig['Claimed'])
insurance['Type'].value_counts().plot(kind='bar')
sns.countplot(insurance_orig['Type'],hue=insurance_orig['Claimed'])

insurance['Channel'].value_counts().plot(kind='bar')
sns.countplot(insurance_orig['Channel'],hue=insurance_orig['Claimed'])

insurance['Product Name'].value_counts().plot(kind='bar')
sns.countplot(insurance_orig['Product Name'],hue=insurance_orig['Claimed'])
insurance['Destination'].value_counts().plot(kind='bar')
sns.countplot(insurance_orig['Destination'],hue=insurance_orig['Claimed'])

##target variable proprotion
insurance['Claimed'].value_counts().plot(kind='bar')
##Proportion of Yes is arounf ~32%

##Scaling is not required for CART /RF tree based models but required for NN based models

##Multicollinearity 
plt.figure(figsize=(12,7))
sns.heatmap(insurance.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()
##Sales and Commision seem to have strong correlation and that is expected as well

##encoding 
#Converting object variables to corresponding categorical codes 
for j in insurance.columns: 
    if insurance[j].dtype == 'object':
        insurance[j] = pd.Categorical(insurance[j]).codes 
        print('Done for Column :',j)


sns.pairplot(insurance)
plt.show()


##Baseline model
##Baseline model -predicting dominant class -68% ACCURACY that we need to beat

##Splitting into train ,test 
all_labels=insurance['Claimed']
all_ind_ds=insurance.drop(["Claimed"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=777)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)


#####CART with default -grid search parameters#####

start_time1 = time.monotonic()
print(start_time1)
param_grid1 = {
    'criterion': ['entropy','gini'],
    'max_depth': [2,4,6,8,10],
    'min_samples_split':range(1,10),
    'min_samples_leaf':range(1,5)
   
}
cart_obj1 = DecisionTreeClassifier(random_state=777)  
cart_mod1=GridSearchCV(estimator=cart_obj1,param_grid=param_grid1,cv=3,scoring=make_scorer(recall_score),verbose=30)
   
cart_mod1.fit(trainset,train_labels)
##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
cart_train_time=timedelta(seconds=end_time1 - start_time1)
print("training time for CART model is ",cart_train_time)

cart_mod1.best_params_
cart_mod1_bst=cart_mod1.best_estimator_
pred_cart_test1 = cart_mod1_bst.predict(testset)
pred_cart_train1=cart_mod1.predict(trainset)       

##Feature importance 
x2=pd.DataFrame(cart_mod1_bst.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x2[0],x2.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()
#Confusion matrix
confusion_matrix(train_labels,pred_cart_train1)

confusion_matrix(test_labels,pred_cart_test1)


#Classification report 
print(classification_report(test_labels,pred_cart_test1))
##Accuracy 
print('Accuracy Score on testset is',round(accuracy_score(test_labels, pred_cart_test1),4)*100,'%')
cart_test_acc=round(accuracy_score(test_labels, pred_cart_test1),4)*100
#print('Accuracy Score on trainset is',round(accuracy_score(train_labels, pred_cart_train1),2)*100,'%')
##Print the Area Under the Curve
print('Area Under the Curve is',round(roc_auc_score(test_labels,cart_mod1.predict_proba(testset)[:,1]),2)*100,'%')

cart_test_recall=round(recall_score(test_labels,pred_cart_test1),4)*100
cart_test_precision=round(precision_score(test_labels,pred_cart_test1),4)*100
cart_test_f1=round(f1_score(test_labels,pred_cart_test1),4)*100
##ROC curve 
probs1 = cart_mod1_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs1 = probs1[:, 1]
auc1 = roc_auc_score(test_labels, probs1)
print('AUC for grid searched model CART: %.3f' % auc1)
# calculate roc curve
fpr1, tpr1, thresholds1 = roc_curve(test_labels, probs1)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr1, tpr1, marker='.')
# show the plot
plt.show()
cart_test_auc=round(auc1,4)*100

###RF Model with gridSearch ################
start_time2 = time.monotonic()
print(start_time2)

param_grid2 = {
    'criterion': ['entropy','gini'],
    'max_depth': [3,4,6],
    'max_features': [0,3,4,6,8,10,12,0.2],
    'min_samples_leaf': [10,50, 100],
    'min_samples_split': [150, 300],
    'n_estimators': [10,15,25,50]
}

rfc_obj2=RandomForestClassifier(random_state=777)
rfc_mod_gs2=GridSearchCV(estimator=rfc_obj2,param_grid=param_grid2,cv=3,scoring=make_scorer(recall_score),verbose=30)
rfc_mod_gs2.fit(trainset,train_labels)

##Capturing training time
end_time2 = time.monotonic()
print(end_time2)
rf_train_time=timedelta(seconds=end_time2 - start_time2)
print("training time for RF model is ",rf_train_time)

print(rfc_mod_gs2.best_params_)
rfc_mod_bst2=rfc_mod_gs2.best_estimator_
pred_test_gs2=rfc_mod_bst2.predict(testset)
pred_train_gs2=rfc_mod_bst2.predict(trainset)
##Confusion matrix
print(confusion_matrix(train_labels,pred_train_gs2))
print(confusion_matrix(test_labels,pred_test_gs2))
##Classification report
print(classification_report(test_labels,pred_test_gs2))
##Accuracy Score
print(accuracy_score(train_labels,pred_train_gs2))
print(accuracy_score(test_labels,pred_test_gs2))
rf_test_acc=round(accuracy_score(test_labels, pred_test_gs2),4)*100
##AUC Score
print('Area Under the Curve is',round(roc_auc_score(test_labels,rfc_mod_bst2.predict_proba(testset)[:,1]),2)*100,'%')


rf_test_recall=round(recall_score(test_labels,pred_test_gs2),4)*100
rf_test_precision=round(precision_score(test_labels,pred_test_gs2),4)*100
rf_test_f1=round(f1_score(test_labels,pred_test_gs2),4)*100

##ROC curve
#print(classification_report(test_labels,pred_test_normal))
# AUC and ROC for the test data
# predict probabilities
probs2 = rfc_mod_bst2.predict_proba(testset)
# keep probabilities for the positive outcome only
probs2 = probs2[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc2 = roc_auc_score(test_labels, probs2)
print('AUC for grid searched model RF: %.3f' % auc2)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr2, tpr2, thresholds2 = roc_curve(test_labels, probs2)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr2, tpr2, marker='.')
# show the plot
plt.show()
rf_test_auc=round(auc2,4)*100

x55=pd.DataFrame(rfc_mod_bst2.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x55[0],x55.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in % For RandomForest model')
plt.title('Feature Importance Plot')
plt.show()

###NNet model ####
##Outlier treatement with ul ll winsorization 
insurancenn=insurance.copy()
cat_col,num_col=get_num_cat_colnames(insurancenn)

insurancenn.boxplot()
plt.xticks(rotation=90);

for k in num_col:
     insurancenn[k]=treat_outlier_ul_ll_winsor(insurancenn[k])
     
insurancenn.boxplot()
plt.xticks(rotation=90);
    
##trainTest Split for NNET

all_labels_nn=insurancenn['Claimed']
all_ind_ds_nn=insurancenn.drop(["Claimed"],axis=1)
# splitting data into training and test set for independent attributes
trainsetnn,testsetnn, train_labelsnn, test_labelsnn = train_test_split(all_ind_ds_nn, all_labels_nn, test_size=.30, random_state=777)
print(trainsetnn.shape)
print(train_labelsnn.shape)
print(testsetnn.shape)
print(test_labelsnn.shape) 
     
##Scaling reqd for neuralnet 
sobj=StandardScaler()
data_scaled_trainsetnn=sobj.fit_transform(trainsetnn)
data_scaled_testsetnn=sobj.transform(testsetnn)



# #data_scaled_trainset.boxplot()
# #plt.xticks(rotation = 90);
start_time3 = time.monotonic()
print(start_time3)

nn_cl=MLPClassifier(random_state=777)
param_grid3 = {
    'hidden_layer_sizes': [250,300,350],
    'max_iter': [500,1000,2000],
    'solver': ['sgd','adam'],
    'tol': [0.01,0.001]
}

nn_mod1 = GridSearchCV(estimator = nn_cl, param_grid = param_grid3, cv = 10,verbose=30,scoring=make_scorer(recall_score))

nn_mod1.fit(data_scaled_trainsetnn,train_labelsnn)

###capturing model training time 
##Capturing training time
end_time3 = time.monotonic()
print(end_time3)
nn_train_time=timedelta(seconds=end_time3 - start_time3)
print("training time for NN model is ",nn_train_time)

nn_mod1.best_params_
nn_mod1_bst=nn_mod1.best_estimator_
pred_train_nn=nn_mod1_bst.predict(data_scaled_trainsetnn)
pred_test_nn=nn_mod1_bst.predict(data_scaled_testsetnn)

print(confusion_matrix(train_labelsnn,pred_train_nn))
print(confusion_matrix(test_labelsnn,pred_test_nn))


print(classification_report(test_labelsnn,pred_test_nn))

print(accuracy_score(train_labels,pred_train_nn))
print(accuracy_score(test_labels,pred_test_nn))
nn_test_acc=round(accuracy_score(test_labels, pred_test_nn),4)*100
nn_test_recall=round(recall_score(test_labels,pred_test_nn),4)*100
nn_test_precision=round(precision_score(test_labels,pred_test_nn),4)*100
nn_test_f1=round(f1_score(test_labels,pred_test_nn),4)*100
# AUC and ROC for the test data
# predict probabilities
probs3 = nn_mod1_bst.predict_proba(data_scaled_testsetnn)
# keep probabilities for the positive outcome only
probs3 = probs3[:, 1]
# calculate AUC
auc3 = roc_auc_score(test_labelsnn, probs3)
print('AUC: %.3f' % auc3)
# calculate roc curve
fpr3, tpr3, thresholds = roc_curve(test_labelsnn, probs3)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.')
# show the plot
plt.show()

nn_test_auc=round(auc3,4)*100

##comparison of results on testset 

index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']
data = pd.DataFrame({
            'CART Test':[cart_test_acc,cart_test_auc,cart_test_recall,cart_test_precision,cart_test_f1],    
        'Random Forest Test':[rf_test_acc,rf_test_auc,rf_test_recall,rf_test_precision,rf_test_f1],      
        'Neural Network Test':[nn_test_acc,nn_test_auc,nn_test_recall,nn_test_precision,nn_test_f1]
        },index=index)
print(round(data,4))


##Plotting ROC curve for comparison
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr1, tpr1,color='red',label="CART")
plt.plot(fpr2, tpr2,color='green',label="RF")
plt.plot(fpr3, tpr3,color='black',label="NN")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right')


