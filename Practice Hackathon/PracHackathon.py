# -*- coding: utf-8 -*-
"""
Created on Sat May 15 14:23:47 2021

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
from sklearn.neighbors import KNeighborsClassifier


def getupperlower_outlier(col):
    sorted(col)
    if col.isnull().values.any()==True:
        Q1,Q3=np.nanpercentile(col,[25,75])
        count=col.isnull().sum()
        print("There are NaN values in the column,which we will ignore-No of NaNs=",count)
    else:
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

emp_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\practice hackathon\Train_Dataset.csv')
trainonly_emp=emp_orig.copy()
print(trainonly_emp.shape)

##Droping all rows with NA/NULL values
trainonly_emp=trainonly_emp.dropna(how='all')


print(trainonly_emp.shape)
print(trainonly_emp.head(5))
print(trainonly_emp.dtypes)
print(trainonly_emp.describe())
print(trainonly_emp.info())
trainonly_emp['Ind']="train"

unktestset_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\practice hackathon\Test_Dataset.csv')
unktestset=unktestset_orig.copy()
print(unktestset.shape)

##Droping all rows with NA/NULL values
unktestset=unktestset.dropna(how='all')
print(unktestset.shape)

##merging train and unknown test set 
unktestset['Attrition']=np.nan
unktestset['Ind']="unknown_test"

frames=[trainonly_emp,unktestset]
emp=pd.concat(frames)


##Univariate analysis -Categorical
print(emp['TravelProfile'].unique())
print(emp['Department'].unique())
print(emp['EducationField'].unique())
print(emp['Gender'].unique())
print(emp['Designation'].unique())

print(emp['MaritalStatus'].unique())
print(emp['Involvement'].unique())
print(emp['WorkLifeBalance'].unique())
print(emp['JobSatisfaction'].unique())
print(emp['OverTime'].unique())

print(emp['ESOPs'].unique())
print(emp['Attrition'].unique())
print(emp['EmployeeID'].unique())
##Continuous -uniq values
print(emp['Age'].unique())
print(emp['HomeToWork'].unique())
print(emp['HourlnWeek'].unique())
print(emp['CurrentProfile'].unique())
print(emp['NumCompaniesWorked'].unique())


print(emp['WorkExperience'].unique())
print(emp['LastPromotion'].unique())
print(emp['SalaryHikelastYear'].unique())
print(emp['MonthlyIncome'].unique())
emp['HourlnWeek'].value_counts()




#Count bar plots for categorical variables 
emp['TravelProfile'].value_counts().plot(kind='bar')
emp['Department'].value_counts().plot(kind='bar')
emp['EducationField'].value_counts().plot(kind='bar')
emp['Gender'].value_counts().plot(kind='bar')
emp['Designation'].value_counts().plot(kind='bar')

emp['MaritalStatus'].value_counts().plot(kind='bar')
emp['Involvement'].value_counts().plot(kind='bar')
emp['WorkLifeBalance'].value_counts().plot(kind='bar')
emp['JobSatisfaction'].value_counts().plot(kind='bar')
emp['OverTime'].value_counts().plot(kind='bar')

emp['ESOPs'].value_counts().plot(kind='bar')

##target
emp['Attrition'].value_counts().plot(kind='bar')

#Histograms for continuous variables
sns.distplot(emp.Age,bins=10)
sns.distplot(emp.HomeToWork,bins=10)
sns.distplot(emp.HourlnWeek,bins=10)
sns.distplot(emp.NumCompaniesWorked,bins=10)
sns.distplot(emp.WorkExperience,bins=10)

sns.distplot(emp.LastPromotion,bins=10)
sns.distplot(emp.CurrentProfile,bins=10)
sns.distplot(emp.SalaryHikelastYear,bins=10)
sns.distplot(emp.MonthlyIncome,bins=10)

##Categorical predictor with Target
sns.countplot(emp['TravelProfile'],hue=emp['Attrition'])
sns.countplot(emp['Department'],hue=emp['Attrition'])
sns.countplot(emp['EducationField'],hue=emp['Attrition'])
sns.countplot(emp['Gender'],hue=emp['Attrition'])
sns.countplot(emp['Designation'],hue=emp['Attrition'])
sns.countplot(emp['MaritalStatus'],hue=emp['Attrition'])
sns.countplot(emp['Involvement'],hue=emp['Attrition'])
sns.countplot(emp['WorkLifeBalance'],hue=emp['Attrition'])
sns.countplot(emp['JobSatisfaction'],hue=emp['Attrition'])
sns.countplot(emp['OverTime'],hue=emp['Attrition'])
sns.countplot(emp['ESOPs'],hue=emp['Attrition'])


##Duplicate analysis 
dups=emp.duplicated()
print("The num of duplicate records are =",dups.sum())
##drop unique variable emp id 
emp.drop('EmployeeID',axis=1,inplace=True)
print(emp.shape)

##outlier analysis
emp.boxplot()
plt.xticks(rotation=90)

data_plot=emp[['Age','HomeToWork','HourlnWeek','SalaryHikelastYear','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']]
fig=plt.figure(figsize=(20,20))
for i in range(0,len(data_plot.columns)):
    ax=fig.add_subplot(3,2,i+1)
    sns.boxplot(data_plot[data_plot.columns[i]])
    plt.tight_layout()
print('Shape before Outliers Treatment',emp.shape)

ll1,ul1=getupperlower_outlier(emp['HourlnWeek'])
out1=emp.loc[(emp['HourlnWeek']>ul1)|(emp['HourlnWeek']<ll1),]
print(out1.shape)

ll2,ul2=getupperlower_outlier(emp['HomeToWork'])
out2=emp.loc[(emp['HomeToWork']>ul2)|(emp['HomeToWork']<ll2),]
print(out2.shape)

ll3,ul3=getupperlower_outlier(emp['NumCompaniesWorked'])
out3=emp.loc[(emp['NumCompaniesWorked']>ul3)|(emp['NumCompaniesWorked']<ll3),]
print(out3.shape)

ll4,ul4=getupperlower_outlier(emp['SalaryHikelastYear'])
out4=emp.loc[(emp['SalaryHikelastYear']>ul4)|(emp['SalaryHikelastYear']<ll4),]
print(out4.shape)

ll5,ul5=getupperlower_outlier(emp['WorkExperience'])
out5=emp.loc[(emp['WorkExperience']>ul5)|(emp['WorkExperience']<ll5),]
print(out5.shape)

ll6,ul6=getupperlower_outlier(emp['LastPromotion'])
out6=emp.loc[(emp['LastPromotion']>ul6)|(emp['LastPromotion']<ll6),]
print(out6.shape)

ll7,ul7=getupperlower_outlier(emp['CurrentProfile'])
out7=emp.loc[(emp['CurrentProfile']>ul7)|(emp['CurrentProfile']<ll7),]
print(out7.shape)

ll8,ul8=getupperlower_outlier(emp['MonthlyIncome'])
out8=emp.loc[(emp['MonthlyIncome']>ul8)|(emp['MonthlyIncome']<ll8),]
print(out8.shape)

##Age has NANs but currently no outliers
ll9,ul9=getupperlower_outlier(emp['Age'])
out9=emp.loc[(emp['Age']>ul9)|(emp['Age']<ll9),]
print(out9.shape)


##Missing and Bad data 
print(emp.isnull().sum())
emp.info()

catcols,numcols=get_num_cat_colnames(emp)
##Imputation ##
##Replacing missing values in categorical with MODE
for i in catcols:
    mode = emp[i].mode()
    print(mode)
    print("No. Of NULL values in this column are",emp[i].isnull().sum())
    emp[i].fillna(mode[0],inplace=True)
##This replaces department/gender/designation NULL values with MODE
emp.isnull().sum()
# ##Replacing with mean -Age as no outliers
# mean_age = emp['Age'].mean()
# print(mean_age)
# print("No. Of NULL values in this column are",emp['Age'].isnull().sum())
# emp['Age'].fillna(mean_age,inplace=True)
# emp.isnull().sum()
##Replacing with median -Continuous cols with outliers
for k in emp[['Age','HomeToWork','HourlnWeek','SalaryHikelastYear','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']]:
    median = emp[k].median()
    print(median)
    print("No. Of NULL values in this column are",emp[k].isnull().sum())
    emp[k].fillna(median,inplace=True)
emp.isnull().sum()    

##Outlier treatment after imputation
emp.boxplot()
plt.xticks(rotation=90)
sns.boxplot(emp['Age'])
sns.boxplot(emp['HomeToWork'])
sns.boxplot(emp['HourlnWeek'])
sns.boxplot(emp['NumCompaniesWorked'])
sns.boxplot(emp['SalaryHikelastYear'])
sns.boxplot(emp['WorkExperience'])
sns.boxplot(emp['LastPromotion'])
sns.boxplot(emp['CurrentProfile'])
sns.boxplot(emp['MonthlyIncome'])


for j in emp[['Age','HomeToWork','HourlnWeek','NumCompaniesWorked','SalaryHikelastYear','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']]:
    emp[j]=treat_outlier_ul_ll_winsor(emp[j])  
    print("Winsorization complete for column=",j)

emp.boxplot()
plt.xticks(rotation=90)

##Scaling -not reqd for CART and RF models

##Multicollinearity 
sns.pairplot(emp)
plt.show()
##Heatmap with masking 
plt.figure(figsize=(12,7))
sns.heatmap(emp.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(emp.corr()) )
plt.show()


##Encoding 
##Replacing bad entries with correct ones 
##Gender-F is actually female -For EXACT MATCH

item1="\\b"+"F"+"\\b"
emp.Gender=emp.Gender.str.replace(item1,'Female')
emp['Gender'].value_counts().plot(kind='bar')

item2="\\b"+"M"+"\\b"
emp.MaritalStatus=emp.MaritalStatus.str.replace(item2, 'Married')
emp['MaritalStatus'].value_counts().plot(kind='bar')

#Converting object variables to corresponding categorical codes 
for j in emp[['Gender','MaritalStatus','TravelProfile','Department','EducationField']]: 
        emp[j] = pd.Categorical(emp[j]).codes 
        print('Done for Column :',j)
        
##Encoding categorical nominal variables to numerical values as per order-label encoding
encobj_desig=OrdinalEncoder(categories=[['Executive' ,'Manager' ,'Senior Manager' ,'AVP' ,'VP']])
emp['Designation']=encobj_desig.fit_transform(emp[["Designation"]])    
emp['Designation'].value_counts().plot(kind='bar')    

##demerging
test_unknown=emp.loc[(emp['Ind']=="unknown_test")]
print(test_unknown.shape)
print(test_unknown['Attrition'].unique())
test_unknown.drop(['Attrition','Ind'],axis=1,inplace=True)
print(test_unknown.shape)
emp_train=emp.loc[(emp['Ind']=="train")]
print(emp_train.shape)
emp_train.drop('Ind',axis=1,inplace=True)
print(emp_train.shape)



# ##Scaling reqd for neuralnet 
# sobj=StandardScaler()
# cols_to_scale=['Age','HomeToWork','HourlnWeek','NumCompaniesWorked','SalaryHikelastYear','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']
# data_to_scale=emp[cols_to_scale]
# data_scaled=sobj.fit_transform(data_to_scale.values)
# data_scale_df=pd.DataFrame(data_scaled,columns=cols_to_scale)
# print(data_scale_df.shape)
# cols_not_scale=['Involvement','WorkLifeBalance','JobSatisfaction','ESOPs','OverTime','Attrition','TravelProfile','Department','EducationField',	'Gender','Designation','MaritalStatus','Ind']
# data_not_scale=emp[cols_not_scale]
# data_scaled_all=pd.concat([data_to_scale,data_not_scale],axis=1)
# ##demerge
# test_unknown_nn=data_scaled_all.loc[(data_scaled_all['Ind']=="unknown_test")]
# print(test_unknown_nn.shape)
# print(test_unknown_nn['Attrition'].unique())
# test_unknown_nn.drop(['Attrition','Ind'],axis=1,inplace=True)
# print(test_unknown_nn.shape)
# emp_train_nn=data_scaled_all.loc[(data_scaled_all['Ind']=="train")]
# print(emp_train_nn.shape)
# emp_train_nn.drop('Ind',axis=1,inplace=True)
# print(emp_train_nn.shape)



##train and validate split
all_labels=emp_train['Attrition']
all_ind_ds=emp_train.drop(["Attrition"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=999)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

# ##For NN
# all_labels_nn=emp_train_nn['Attrition']
# all_ind_ds_nn=emp_train_nn.drop(["Attrition"],axis=1)
# # splitting data into training and test set for independent attributes
# trainsetnn,testsetnn, train_labelsnn, test_labelsnn = train_test_split(all_ind_ds_nn, all_labels_nn, test_size=.30, random_state=999)
# print(trainsetnn.shape)
# print(train_labelsnn.shape)
# print(testsetnn.shape)
# print(test_labelsnn.shape)
trainsetnn=trainset.copy()
train_labelsnn=train_labels.copy()
testsetnn=testset.copy()
test_labelsnn=test_labels.copy()

print(trainsetnn.shape)
print(train_labelsnn.shape)
print(testsetnn.shape)
print(test_labelsnn.shape) 
     
##Scaling reqd for neuralnet 
sobj=StandardScaler()
data_scaled_trainsetnn=sobj.fit_transform(trainsetnn)
data_scaled_testsetnn=sobj.fit_transform(testsetnn)
data_scaled_test_unknown=sobj.fit_transform(test_unknown)


##Logistic regression 
model_logis = LogisticRegression(solver='newton-cg',max_iter=10000,verbose=True,n_jobs=2)
model_logis.fit(trainset,train_labels)

pred_train_log=model_logis.predict(trainset) 
print(confusion_matrix(train_labels,pred_train_log))
log_acc_train=accuracy_score(train_labels,pred_train_log)
print("Accuracy on trainset =",accuracy_score(train_labels,pred_train_log))
pred_test_log=model_logis.predict(testset)
print(confusion_matrix(test_labels,pred_test_log))
log_acc_test=accuracy_score(test_labels,pred_test_log)
print(log_acc_test)
print("Accuracy on Validateset=",accuracy_score(test_labels,pred_test_log)) 
##Auc train
probs_train = model_logis.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_train = probs_train[:, 1]
# calculate AUC
auc_train = roc_auc_score(train_labels, probs_train)
print('AUC: %.3f' % auc_train)
##Auc test
probs_test = model_logis.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_test = probs_test[:, 1]
# calculate AUC
auc_test = roc_auc_score(test_labels, probs_test)
print('AUC: %.3f' % auc_test)

###RF Model with gridSearch ################
start_time2 = time.monotonic()
print(start_time2)

param_grid2 = {
    'criterion': ['entropy','gini'],
    'max_depth': [15,20],
    'max_features': [10,20],
    'min_samples_leaf': [10,20],
    'min_samples_split': [10,100],
    'n_estimators': [700]
}

rfc_obj2=RandomForestClassifier(random_state=999)
rfc_mod_gs2=GridSearchCV(estimator=rfc_obj2,param_grid=param_grid2,cv=3,verbose=30)
rfc_mod_gs2.fit(trainset,train_labels)

##Capturing training time
end_time2 = time.monotonic()
print(end_time2)
rf_train_time=timedelta(seconds=end_time2 - start_time2)
print("training time for RF model is ",rf_train_time)

print(rfc_mod_gs2.best_params_)
rfc_mod_bst2=rfc_mod_gs2.best_estimator_
##trainset
pred_train_gs2=rfc_mod_bst2.predict(trainset)
print(confusion_matrix(train_labels,pred_train_gs2))
rf_acc_score_train=accuracy_score(train_labels,pred_train_gs2)
print("Trainset accuracy for RF model=",rf_acc_score_train)
print(accuracy_score(train_labels,pred_train_gs2))
##Testset
pred_test_gs2=rfc_mod_bst2.predict(testset)
print(confusion_matrix(test_labels,pred_test_gs2))
rf_acc_score_test=accuracy_score(test_labels,pred_test_gs2)
print("validate Set Accuracy for RF model=",rf_acc_score_test)
print(accuracy_score(test_labels,pred_test_gs2))

##AUC Score
rf_auc_train=round(roc_auc_score(train_labels,rfc_mod_bst2.predict_proba(trainset)[:,1]),2)*100
print('Area Under the Curve for TrainSet is',round(roc_auc_score(train_labels,rfc_mod_bst2.predict_proba(trainset)[:,1]),2)*100,'%')
rf_auc_test=round(roc_auc_score(test_labels,rfc_mod_bst2.predict_proba(testset)[:,1]),2)*100
print('Area Under the Curve for VALIDATE SET is',round(roc_auc_score(test_labels,rfc_mod_bst2.predict_proba(testset)[:,1]),2)*100,'%')

#####CART with default -grid search parameters#####

start_time1 = time.monotonic()
print(start_time1)
param_grid1 = {
    'criterion': ['entropy','gini'],
    'max_depth': [10,12,14,20,30,50],
    'min_samples_split':[40,30,20,10,5,3,2],
    'min_samples_leaf':[5,4,3,2,1]
   
}
cart_obj1 = DecisionTreeClassifier(random_state=999)  
cart_mod1=GridSearchCV(estimator=cart_obj1,param_grid=param_grid1,cv=10,verbose=30)
   
cart_mod1.fit(trainset,train_labels)
##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
cart_train_time=timedelta(seconds=end_time1 - start_time1)
print("training time for CART model is ",cart_train_time)

cart_mod1.best_params_
cart_mod1_bst=cart_mod1.best_estimator_
##testset
pred_cart_train1=cart_mod1_bst.predict(trainset)   
confusion_matrix(train_labels,pred_cart_train1)
cart_train_acc=round(accuracy_score(train_labels, pred_cart_train1),4)*100
print("Acc on trainset CART MODEL is",cart_train_acc)
print(accuracy_score(train_labels, pred_cart_train1))
probs_cart_train = cart_mod1_bst.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs_cart_train = probs_cart_train[:, 1]
auc1 = roc_auc_score(train_labels, probs_cart_train)
print('AUC for trainset CART: %.3f' % auc1)
##Testset
pred_cart_test1 = cart_mod1_bst.predict(testset)    
confusion_matrix(test_labels,pred_cart_test1)
cart_test_acc=round(accuracy_score(test_labels, pred_cart_test1),4)*100
print("Acc on validate set CART MODEL is",cart_test_acc)
print(accuracy_score(test_labels, pred_cart_test1))
probs_cart_test = cart_mod1_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs_cart_test = probs_cart_test[:, 1]
auc2 = roc_auc_score(test_labels, probs_cart_test)
print('AUC for trainset CART: %.3f' % auc2)


##Neural Net 




# #data_scaled_trainset.boxplot()
# #plt.xticks(rotation = 90);
start_time3 = time.monotonic()
print(start_time3)

nn_cl=MLPClassifier(random_state=999)
param_grid3 = {
    'hidden_layer_sizes': [1024],
    'max_iter': [50],
    'solver': ['sgd','adam'],
    'tol': [0.001],
    'learning_rate':['constant'],
    'learning_rate_init':[0.01]
    
    
}

nn_mod1 = GridSearchCV(estimator = nn_cl, param_grid = param_grid3, cv = 5,verbose=20,n_jobs=-1)

nn_mod1.fit(data_scaled_trainsetnn,train_labelsnn)

###capturing model training time 
##Capturing training time
end_time3 = time.monotonic()
print(end_time3)
nn_train_time=timedelta(seconds=end_time3 - start_time3)
print("training time for NN model is ",nn_train_time)

print("The best parameters found via gridsearch for NN Model are =",nn_mod1.best_params_)
nn_mod1_bst=nn_mod1.best_estimator_
##Trainset
pred_train_nn=nn_mod1_bst.predict(data_scaled_trainsetnn)
print(confusion_matrix(train_labelsnn,pred_train_nn))
acc_nn_train=accuracy_score(train_labelsnn,pred_train_nn)
print(acc_nn_train)
print("Accuracy on the Train set  =",accuracy_score(train_labelsnn,pred_train_nn))
probs4 = nn_mod1_bst.predict_proba(data_scaled_trainsetnn)
# keep probabilities for the positive outcome only
probs4 = probs4[:, 1]
# calculate AUC
auc4 = roc_auc_score(train_labelsnn, probs4)
print('AUC: %.3f' % auc4)


pred_test_nn=nn_mod1_bst.predict(data_scaled_testsetnn)

print(confusion_matrix(test_labelsnn,pred_test_nn))
acc_nn_test=accuracy_score(test_labelsnn,pred_test_nn)
print(acc_nn_test)
print("Accuracy on the VALIDATE set  =",accuracy_score(test_labelsnn,pred_test_nn))
probs3 = nn_mod1_bst.predict_proba(data_scaled_testsetnn)
# keep probabilities for the positive outcome only
probs3 = probs3[:, 1]
# calculate AUC
auc3 = roc_auc_score(test_labelsnn, probs3)
print('AUC on VALIDATE set: %.3f' % auc3)

##Accuracy on Validate set 
index=['Accuracy', 'AUC']
dataf = pd.DataFrame({'Logistic Train':[log_acc_train,auc_train],
        'Logistic Test':[log_acc_test,auc_test],
       'RF Train':[rf_acc_score_train,rf_auc_train],
        'RF Test':[rf_acc_score_test,rf_auc_test],
        'CART Train':[cart_train_acc,auc1],
        'CART Test':[cart_test_acc,auc2],
        'NN Train':[acc_nn_train,auc4],
        'NN Test':[acc_nn_test,auc3]},index=index)
round(dataf,5)

##Testset processing 
print(unktestset_orig.shape)
id=unktestset['EmployeeID']
print(test_unknown.shape)
print(len(data_scaled_test_unknown))

res_cols=['EmployeeID','Attrition']
id=id.to_numpy()

pred_unk_cart=cart_mod1_bst.predict(test_unknown)
pred_unk_nn=nn_mod1_bst.predict(data_scaled_test_unknown)

combine=np.vstack((id,pred_unk_cart)).T
combine_nn=np.vstack((id,pred_unk_nn)).T

result_cart=pd.DataFrame(data=combine,columns=res_cols)
result_nn=pd.DataFrame(data=combine_nn,columns=res_cols)
result_cart.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\practice hackathon\result_cart.csv',index=False)
result_nn.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\practice hackathon\result_nn.csv',index=False)








