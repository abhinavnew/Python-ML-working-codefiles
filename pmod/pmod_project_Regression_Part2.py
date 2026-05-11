# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:13:07 2021

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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


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


# def select_badvalues_rows(dframe,cat_colnames,bad_vals):
#     smalldfs=[]
#     for i in cat_colnames :           
#         smalldfs.append(dframe.loc[dframe[i].str.contains('|'.join(bad_values))==True])
#         print("For column name =",i)
#     print(type(smalldfs))
#     largedf=pd.concat(smalldfs,ignore_index=True)    
#     return largedf

# c=select_badvalues_rows(df1,cat_col,bad_values)
# cat_col,num_col=get_num_cat_colnames(df1)
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 30)

pd.set_option("display.max_colwidth",30)
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

holiday_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\Logistic_linReg_LDA_Project\holiday_package.csv')
holiday=holiday_orig.copy()
holiday.drop(holiday.columns[holiday.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


print(holiday.shape)
print(holiday.head(5))
print(holiday.dtypes)
print(holiday.describe())
print(holiday.info())

##Univariate analysis -Categorical
print(holiday['educ'].unique())
print(holiday['foreign'].unique())

#Count bar plots for categorical variables 
holiday['educ'].value_counts().plot(kind='bar')

holiday['foreign'].value_counts().plot(kind='bar')

holiday['Holliday_Package'].value_counts().plot(kind='bar')
holiday['no_young_children'].value_counts().plot(kind='bar')
holiday['no_older_children'].value_counts().plot(kind='bar')
#Histograms for continuous variables
sns.distplot(holiday.Salary,bins=10)
sns.distplot(holiday.age,bins=10)
sns.distplot(holiday.no_young_children,bins=4)
sns.distplot(holiday.no_older_children,bins=4)

##Checking skewness for symmetry 
print(holiday['Salary'].skew())
print(holiday['age'].skew())
print(holiday['no_young_children'].skew())
print(holiday['no_older_children'].skew())



##Bivariate analysis 
sns.pairplot(holiday)
plt.show()

##Categorical predictor with Target
sns.countplot(holiday['educ'],hue=holiday['Holliday_Package'])
sns.countplot(holiday['foreign'],hue=holiday['Holliday_Package'])
sns.countplot(holiday['no_young_children'],hue=holiday['Holliday_Package'])
sns.countplot(holiday['no_older_children'],hue=holiday['Holliday_Package'])



##Heatmap with masking 

plt.figure(figsize=(12,7))
sns.heatmap(holiday.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(holiday.corr()) )
plt.show()
##Duplicates 
dups=holiday.duplicated()
print("The num of duplicate records are =",dups.sum())

##outlier analysis
holiday.boxplot()
plt.xticks(rotation=90)
ll1,ul1=getupperlower_outlier(holiday['Salary'])
out1=holiday.loc[(holiday['Salary']>ul1)|(holiday['Salary']<ll1),]
print(out1)
##Outlier treatement -as they will disturb line of best fit
holiday['Salary']=treat_outlier_ul_ll_winsor(holiday['Salary'])
holiday.boxplot()
plt.xticks(rotation=90)

# for j in emp[['Age','PercentSalaryIncrement','YearsSinceLastPromotion','TotalWorkExp','MonthlySalary','YearsWithCurrManager','DistanceFromHome','NumCompaniesWorked']]:
#     emp[j]=treat_outlier_1_99_winsor(emp[j])  
#     print("Winsorization complete for column=",j)

##Missing values /bad values
holiday.isnull().sum()
holiday.info()
##Any constant columns
d= return_constant_columns(holiday)
print(d)

##Scaling -Not required 

##Multicollinearity -Checked and not much hence not dropping any columns

##Encoding 
##Educ -already coded as per order 
##Need to encode Holliday_Pacakage and Foreigner ,both are binary 

#Converting object variables to corresponding categorical codes 
for j in holiday.columns: 
    if holiday[j].dtype == 'object':
        holiday[j] = pd.Categorical(holiday[j]).codes 
        print('Done for Column :',j)
        
      
 ##train test split 
# Copy all the predictor variables into X dataframe
all_ind_ds= holiday.drop('Holliday_Package', axis=1)
# Copy target into the y dataframe. 
all_labels = holiday[['Holliday_Package']]

# Split X and y into training and test set in 75:25 ratio

trainset, testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=0.30 , random_state=888)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

trainsetnn=trainset.copy()
train_labelsnn=train_labels.copy()
testsetnn=testset.copy()
test_labelsnn=test_labels.copy()
       
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


##LDA 

lda_classifier_obj = LinearDiscriminantAnalysis()
lda_mod=lda_classifier_obj.fit(trainset,train_labels)
pred_train_lda=lda_mod.predict(trainset)
pred_test_lda=lda_mod.predict(testset)

##train results

print(confusion_matrix(train_labels,pred_train_lda))
lda_acc_train=accuracy_score(train_labels,pred_train_lda)
print(lda_acc_train)
print(accuracy_score(train_labels,pred_train_lda)) 

probs_train_lda = lda_mod.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train_lda = probs_train_lda[:, 1]
# calculate AUC
auc_train_lda = roc_auc_score(train_labels, probs_train_lda)
print('AUC: %.3f' % auc_train_lda)
# calculate roc curve
train_fpr_lda, train_tpr_lda, train_thresholds_lda = roc_curve(train_labels, probs_train_lda)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(train_fpr_lda, train_tpr_lda)

##Test set
print(confusion_matrix(test_labels,pred_test))
lda_acc_test=accuracy_score(test_labels,pred_test_lda)
print(lda_acc_test)
print(accuracy_score(test_labels,pred_test_lda))

 
probs_test_lda = lda_mod.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test_lda = probs_test_lda[:, 1]
# calculate AUC
auc_test_lda = roc_auc_score(test_labels, probs_test_lda)
print('AUC: %.3f' % auc_test_lda)
# calculate roc curve
test_fpr_lda, test_tpr_lda, test_thresholds_lda = roc_curve(test_labels, probs_test_lda)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(test_fpr, test_tpr)


index=['Accuracy', 'AUC']
data = pd.DataFrame({'Logistic Train':[log_acc_train,auc_train],
        'Logistic Test':[log_acc_test,auc_test],
       'LDA Train':[lda_acc_train,auc_train_lda],
        'LDA Test':[lda_acc_test,auc_test_lda]},index=index)
round(data,4)


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


##trainTest Split for NNET


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
    'hidden_layer_sizes': [(25,25),(15,15)],
    'max_iter': [50,75,100,125],
    'solver': ['sgd','adam'],
    'tol': [0.01,0.001],
    'learning_rate':['constant','adaptive']
    
}

nn_mod1 = GridSearchCV(estimator = nn_cl, param_grid = param_grid3, cv = 10,verbose=30)

nn_mod1.fit(data_scaled_trainsetnn,train_labelsnn)

###capturing model training time 
##Capturing training time
end_time3 = time.monotonic()
print(end_time3)
nn_train_time=timedelta(seconds=end_time3 - start_time3)
print("training time for NN model is ",nn_train_time)

print("The best parameters found via gridsearch are =",nn_mod1.best_params_)
nn_mod1_bst=nn_mod1.best_estimator_
pred_train_nn=nn_mod1_bst.predict(data_scaled_trainsetnn)
pred_test_nn=nn_mod1_bst.predict(data_scaled_testsetnn)

print(accuracy_score(train_labels,pred_train_nn))
print("Accuracy on the Test set of HolidayDS =",accuracy_score(test_labels,pred_test_nn))

print(confusion_matrix(train_labelsnn,pred_train_nn))
print(confusion_matrix(test_labelsnn,pred_test_nn))


print(classification_report(test_labelsnn,pred_test_nn))


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

##Trainset
probs4 = nn_mod1_bst.predict_proba(data_scaled_trainsetnn)
# keep probabilities for the positive outcome only
probs4 = probs4[:, 1]
# calculate AUC
auc4 = roc_auc_score(train_labelsnn, probs4)
print('AUC: %.3f' % auc4)
# calculate roc curve
fpr3, tpr3, thresholds = roc_curve(test_labelsnn, probs3)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.')
# show the plot
plt.show()

nn_test_auc=round(auc3,4)*100








