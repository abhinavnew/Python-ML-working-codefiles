# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:59:56 2021

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
import xgboost as xgb
from xgboost import plot_importance
import pickle
from imblearn.over_sampling import SMOTE 
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
print("merged dataset dimensions are",emp.shape)


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


emp['HourlnWeek'].value_counts().plot(kind='bar')


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
##emp.boxplot()

print('Shape before Outliers Treatment',emp.shape)



##Age has NANs but currently no outliers
ll9,ul9=getupperlower_outlier(emp['Age'])
out9=emp.loc[(emp['Age']>ul9)|(emp['Age']<ll9),]
print(out9.shape)


##Missing and Bad data 
print(emp.isnull().sum())
emp.info()


##Imputation ##
catcols,numcols=get_num_cat_colnames(emp)
##Replacing missing values in categorical with MODE
for i in catcols:
    mode = emp[i].mode()
    print(mode)
    print("No. Of NULL values in this column are",emp[i].isnull().sum())
    emp[i].fillna(mode[0],inplace=True)
# ##This replaces department/gender/designation NULL values with MODE
emp.isnull().sum()
# # ##Replacing with mean -Age as no outliers
mean_age = emp['Age'].mean()
print(mean_age)
print("No. Of NULL values in this column are",emp['Age'].isnull().sum())
emp['Age'].fillna(mean_age,inplace=True)
emp.isnull().sum()
# ##Replacing with median -Continuous cols with outliers
for k in emp[['Age','HomeToWork','HourlnWeek','SalaryHikelastYear','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']]:
    median = emp[k].median()
    print(median)
    print("No. Of NULL values in this column are",emp[k].isnull().sum())
    emp[k].fillna(median,inplace=True)
emp.isnull().sum()    

##Outlier treatment after imputation
##emp.boxplot()
plt.xticks(rotation=90);
sns.boxplot(emp['Age'])
sns.boxplot(emp['HomeToWork'])
sns.boxplot(emp['HourlnWeek'])
sns.boxplot(emp['NumCompaniesWorked'])
sns.boxplot(emp['SalaryHikelastYear'])
sns.boxplot(emp['WorkExperience'])
sns.boxplot(emp['LastPromotion'])
sns.boxplot(emp['CurrentProfile'])
sns.boxplot(emp['MonthlyIncome'])

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

big=out8.loc[(out8['MonthlyIncome']>36000),]

####OUTLIER TREATMENT produces best results:tried ul,ll 1,99 and 5,95##############
for j in emp[['HomeToWork','WorkExperience','LastPromotion','CurrentProfile','MonthlyIncome']]:
     emp[j]=treat_outlier_1_99_winsor(emp[j])  
     print("Winsorization complete for column=",j)
     
for i in emp[['HourlnWeek','NumCompaniesWorked','SalaryHikelastYear']]:
     emp[i]=treat_outlier_ul_ll_winsor(emp[i])  
     print("Winsorization complete for column=",i)

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

#Converting object variables to corresponding Dummy/One HOT encoding 
# for j in emp[['Gender','MaritalStatus','TravelProfile','Department','EducationField']]: 
#         emp[j] = pd.Categorical(emp[j]).codes 
#         print('Done for Column :',j)


dummies=pd.get_dummies(emp[["MaritalStatus", "Gender","TravelProfile","Department","EducationField"]],
                       columns=["MaritalStatus", "Gender","TravelProfile","Department","EducationField"],
                       prefix=["STATUS", "GENDER","TRAVEL","DEP","EDUC"],
                       drop_first=True)
col_to_rem=["MaritalStatus","Gender","TravelProfile","Department","EducationField"]
emp = pd.concat([emp, dummies], axis=1)
emp.drop(col_to_rem,axis = 1,inplace=True)
print(emp.shape)
        
##Encoding categorical nominal variables to numerical values as per order-label encoding
encobj_desig=OrdinalEncoder(categories=[['Executive' ,'Manager' ,'Senior Manager' ,'AVP' ,'VP']])
emp['Designation']=encobj_desig.fit_transform(emp[["Designation"]])    
emp['Designation'].value_counts().plot(kind='bar')  

  

##demerging
all_data_no_target=emp.drop(['Attrition','Ind'],axis=1)
allscaleobj=StandardScaler()
data_scaled_all=allscaleobj.fit_transform(all_data_no_target.values)
data_scaled_all_df=pd.DataFrame(data=data_scaled_all,columns=all_data_no_target.columns)
sel_cols=emp[['Attrition','Ind']]
sel_cols=sel_cols.reset_index()
frames2=[data_scaled_all_df,sel_cols]
nn_scaled=pd.concat([data_scaled_all_df,sel_cols],axis=1)
nn_scaled.drop(['index'],axis=1,inplace=True)
print("merged dataset dimensions are",nn_scaled.shape)


test_unknown=emp.loc[(emp['Ind']=="unknown_test")]
print(test_unknown.shape)
print(test_unknown['Attrition'].unique())
test_unknown.drop(['Attrition','Ind'],axis=1,inplace=True)
print(test_unknown.shape)
emp_train=emp.loc[(emp['Ind']=="train")]
print(emp_train.shape)
emp_train.drop('Ind',axis=1,inplace=True)
print(emp_train.shape)

test_unknown_nn=nn_scaled.loc[(nn_scaled['Ind']=="unknown_test")]
print(test_unknown_nn.shape)
print(test_unknown_nn['Attrition'].unique())
test_unknown_nn.drop(['Attrition','Ind'],axis=1,inplace=True)
print(test_unknown_nn.shape)
emp_train_nn=nn_scaled.loc[(nn_scaled['Ind']=="train")]
print(emp_train_nn.shape)
emp_train_nn.drop('Ind',axis=1,inplace=True)
print(emp_train_nn.shape)


##train and validate split
all_labels=emp_train['Attrition']
all_ind_ds=emp_train.drop(["Attrition"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=123)
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


##Preparing trainset for xgboost

# dtrain=xgb.DMatrix(trainset.values,train_labels.values)
# print("Type of xgb train set is = ",type(dtrain))
# dval=xgb.DMatrix(testset.values,test_labels.values)
# print("Type of xgb train set is =",type(dval))

# param={'objective': 'binary:logistic',
#        'booster':'gbtree',
#        'max_depth':[5,8,10],
#        'min_child_weight':1,
#        'eval_metric':'error',
#        'eta':0.01
#               }

# 'lambda':[1,2,5],
#        'lambda_bias':0,
#        'alpha':[0,2]

#print("Type of param is =",type(param))  
 
##Train a xgboost model 
start_time1=time.monotonic()
print(start_time1)
from xgboost import XGBClassifier
# bst_xgb_mod1=xgb.train(params=param,
#                        dtrain=dtrain,
#                        num_boost_round=999,
#                        early_stopping_rounds=10,
#                        evals=(dval,"validationset"),
#                        verbose_eval=True)

xgbobj=XGBClassifier(random_state=123)
eval_set=[(testset,test_labels)]

param_grid = {
    'n_estimators': [750],
    'max_depth': [4,5,6],
    #'min_samples_leaf': [1,2,3],
    #'min_samples_split': [2,3],
    'eta':[0.01,0.1,0.2],
    'early_stopping_rounds':[50],
    'subsample':[0.8],
    'Lambda':[0.5,2,5],
    'alpha':[0.2]
    
       
}


xgb_mod=GridSearchCV(estimator=xgbobj,param_grid=param_grid,cv=6,verbose=20)
xgb_mod.fit(trainset,train_labels)
##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
train_time=timedelta(seconds=end_time1 - start_time1)
print("training time of xgb model= ",train_time)
print("Best params are =",xgb_mod.best_params_)
print("Best score after gridsearch =",xgb_mod.best_score_)
xgb_mod_bst=xgb_mod.best_estimator_

##make predictions

pred_train=xgb_mod_bst.predict(trainset)
print(pred_train.size)
print("Accuracy on TRAIN set =",accuracy_score(pred_train,train_labels))


pred_val=xgb_mod_bst.predict(testset)
print(pred_val.size)
print("Accuracy on VALIDATE set= ",accuracy_score(pred_val,test_labels))

pred_whole=xgb_mod_bst.predict(all_ind_ds)
print(pred_whole.size)
print("Accuracy on WHOLE TRAIN set= ",accuracy_score(pred_whole,all_labels))

##Importance 
plot_importance(xgb_mod_bst)
##WHOLE TRAIN
xgb_mod2=xgb_mod_bst.fit(all_ind_ds,all_labels)
pred2=xgb_mod2.predict(all_ind_ds)
print(pred2.size)
print("accuracy on best model BUT fitted to all training data =",accuracy_score(pred2,all_labels))


# ##Trainset

probs_xgb = xgb_mod2.predict_proba(all_ind_ds)
# keep probabilities for the positive outcome only
probs_xgb = probs_xgb[:, 1]
# calculate AUC
auc_xgb = roc_auc_score(all_labels, probs_xgb)
print('AUC: %.3f' % auc_xgb)
# ##Calc ROC curve and threshold 
# fpr4, tpr4, thresholds_xgb = roc_curve(all_labels, probs_xgb)
# plt.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr4, tpr4, marker='.')
# # show the plot
# plt.show()

# ##optimal threshold 

for j in np.arange(0.1,1,0.1):
    custom_prob = j #defining the cut-off value of our choice
    custom_cutoff_data=[]#defining an empty list
    for i in range(0,len(all_labels)):#defining a loop for the length of the train data
        if np.array(probs_xgb)[i] > custom_prob:#issuing a condition for our probability values to be 
            #greater than the custom cutoff value
            a=1#if the probability values are greater than the custom cutoff then the value should be 1
        else:
            a=0#if the probability values are less than the custom cutoff then the value should be 0
        custom_cutoff_data.append(a)#adding either 1 or 0 based on the condition to the end of the list defined by us
    print(round(custom_prob,8),metrics.accuracy_score(all_labels,custom_cutoff_data))




# print(accuracy_score(pred2,test_labels))



# probs_nn_unk = xgb_mod2.predict_proba(test_unknown)
# probs_nn_unk = probs_nn_unk[:, 1]
# data_pred_custom_cutoff_unk=[]
# for i in range(0,len(probs_nn_unk)):
#     if np.array(probs_nn_unk)[i]>0.425:
#         a=1
#     else:
#         a=0
#     data_pred_custom_cutoff_unk.append(a)

# pred_unk_nn=pd.Series(data_pred_custom_cutoff_unk,name="Attrition")


print(unktestset_orig.shape)
id=unktestset['EmployeeID']
print(test_unknown.shape)
print("making predictions on BEST model fitted over COMPLETE dataset ")
pred_unk=xgb_mod2.predict(test_unknown)
pred_unk=pd.Series(pred_unk.astype(int),name="Attrition")

result_xgb=pd.concat([id,pred_unk],axis=1)
print(result_xgb.shape)

##Dynamic file name with datetime stamp
snapshotdate = datetime.datetime.today().strftime('%d-%m-%Y_%H_%M_%S')
print(snapshotdate)
myfile_name='thresh_xgb_'+snapshotdate+'.csv'
print(myfile_name)
p='E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\practice hackathon\\submission files\\'
result_xgb.to_csv(p+myfile_name,index=False)

#  ##Save the trained model on disk 
# pickle.dump(xgb_mod2,open("xgbmod_1Jun_99.2Full_Smotte.dat","wb"))
# print("saved model to :xgbmod_1Jun_99.2Full_Smotte.dat")

# # loaded_xgb_mod_disk=pickle.load(open("xgbmod1_withpickle.dat","rb"))
# # print("Loaded from disk : loaded_xgb_mod_disk-from file xgbmod1_withpickle.dat ")

# ##saving on colab 
# path=''
# pickle_out=open(path+"mod.dat","wb")
# pickle.dump(mod,pickle_out)
# pickle_out.close()

