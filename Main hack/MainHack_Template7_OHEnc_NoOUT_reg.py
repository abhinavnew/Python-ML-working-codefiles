# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 07:22:20 2021

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
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_auc_score,roc_curve
 ##for clustering
import time
from datetime import timedelta
from sklearn.cluster import KMeans  ## for k means
#from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,accuracy_score
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
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import plot_importance
import pickle


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

####Read file
emp_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\real hackathon\Train_Data_ET.csv')
trainonly_emp=emp_orig.copy()
print(trainonly_emp.shape)

####Droping all rows with NA/NULL values
trainonly_emp=trainonly_emp.dropna(how='all')

#ID-unique
#Age-continuous 
#recently_upskilled-Categorical binary 
#region-Categorical -text 
#RemoteWork-Categorical binary
#%salary Increment -continuous 
#Office hours -continuous 
#stockoption level-Categorical binary 
#College tier-Categorical -text 
#YearsSinceLastPromotion-Continuous -numerical
#HighestEducation-Categorical -text object 
#Businesstravel-Categorical -text object 
#Joblevel-Categorical -numerical encoded
#Marital status -Categorical -text object 
#Monthly salary -Continuous 
#yof joining -Categorical
#TotalWorkExp-continuous -numerical
#Gender -Categorical -text object 
#YearswithCurrManager-continuous 
#DistanceFrmHome-Continuous 
#NumCompaniesWorked -continuous
#TrainingingTimesLastYear -continuous 
#Dept-Categorical -text 
#Envsat1-categorical -encoded already 
#EnvSat2-Categorical -encoded already 
#Jobsat1 -encoded
#job sat2 -categorical -encoded
#Job invol1-Cat-encoded 
#Job involv2 -cat -encoded 
#Performancerating1 -Cat-encoded
#Performrating2-Cat-encoded
#EmpTurnover-target

print(trainonly_emp.shape)
print(trainonly_emp.columns)
print(trainonly_emp.head(5))
print(trainonly_emp.dtypes)
print(trainonly_emp.describe())
print(trainonly_emp.info())
trainonly_emp['Ind']="train"

unktestset_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\real hackathon\Test_Data_ET.csv')
unktestset=unktestset_orig.copy()
print(unktestset.shape)

##Droping all rows with NA/NULL values
unktestset=unktestset.dropna(how='all')
print(unktestset.shape)

####Merging train and unknown test set 
unktestset['Employeeturnover']=np.nan
unktestset['Ind']="unknown_test"

frames=[trainonly_emp,unktestset]
emp=pd.concat(frames)
print("merged dataset dimensions are",emp.shape)

##drop unique variable emp id 
print(emp['ID'].nunique())

emp.drop('ID',axis=1,inplace=True)
print(emp.shape)


####Univariate analysis -Categorical
print(emp['Recently_upskilled'].unique())
print(emp['Region'].unique())
print(emp['RemoteWork'].unique())
print(emp['StockOptionLevel'].unique())
print(emp['College_Tier'].unique())#has nan

print(emp['HighestEducation'].unique())
print(emp['JobLevel'].unique())

print(emp['BusinessTravel'].unique())
print(emp['MaritalStatus'].unique())
print(emp['yof joining'].unique())
print(emp['Gender'].unique())
print(emp['Department'].unique())

print(emp['EnvironmentSatisfaction1'].unique())##has nan
print(emp['EnvironmentSatisfaction2'].unique())##has nan
print(emp['JobSatisfaction1'].unique())#has nan
print(emp['JobSatisfaction2'].unique())##has nan
print(emp['JobInvolvement1'].unique())#has nan
print(emp['JobInvolvement2'].unique())#has nan
print(emp['PerformanceRating1'].unique())#has nan
print(emp['PerformanceRating2'].unique())#has nan
print(emp['Employeeturnover'].unique())
print(emp['Employeeturnover'].value_counts())##Target class around 20% Vs 80% 

##Continuous -uniq values
print(emp['Age'].unique())
print(emp['PercentSalaryIncrement'].unique())
print(emp['OfficeHours'].unique())
print(emp['YearsSinceLastPromotion'].unique())#has nan
print(emp['MonthlySalary'].unique())


print(emp['TotalWorkExp'].unique())
print(emp['YearsWithCurrManager'].unique())
print(emp['DistanceFromHome'].unique())
print(emp['NumCompaniesWorked'].unique())#has nan
print(emp['TrainingTimesLastYear'].unique())


##Count bar plots for categorical variables 

(emp['Recently_upskilled'].value_counts().plot(kind='bar'))
(emp['Region'].value_counts().plot(kind='bar'))
(emp['RemoteWork'].value_counts().plot(kind='bar'))
(emp['StockOptionLevel'].value_counts().plot(kind='bar'))
(emp['College_Tier'].value_counts().plot(kind='bar'))#has nan
(emp['HighestEducation'].value_counts().plot(kind='bar'))

(emp['BusinessTravel'].value_counts().plot(kind='bar'))
(emp['MaritalStatus'].value_counts().plot(kind='bar'))
(emp['yof joining'].value_counts().plot(kind='bar'))
(emp['Gender'].value_counts().plot(kind='bar'))
(emp['Department'].value_counts().plot(kind='bar'))

(emp['EnvironmentSatisfaction1'].value_counts().plot(kind='bar'))##has nan
(emp['EnvironmentSatisfaction2'].value_counts().plot(kind='bar'))##has nan
(emp['JobSatisfaction1'].value_counts().plot(kind='bar'))#has nan
(emp['JobSatisfaction2'].value_counts().plot(kind='bar'))##has nan
(emp['JobInvolvement1'].value_counts().plot(kind='bar'))#has nan
(emp['JobInvolvement2'].value_counts().plot(kind='bar'))#has nan
(emp['PerformanceRating1'].value_counts().plot(kind='bar'))#has nan
(emp['PerformanceRating2'].value_counts().plot(kind='bar'))#has nan

##target analysis
(emp['Employeeturnover'].value_counts().plot(kind='bar'))


##Histograms for continuous variables
sns.distplot(emp.Age,bins=10)
sns.distplot(emp.PercentSalaryIncrement,bins=10)
sns.distplot(emp.OfficeHours,bins=10)
sns.distplot(emp.YearsSinceLastPromotion,bins=10)
sns.distplot(emp.MonthlySalary,bins=10)

sns.distplot(emp.TotalWorkExp,bins=10)
sns.distplot(emp.YearsWithCurrManager,bins=10)
sns.distplot(emp.DistanceFromHome,bins=10)
sns.distplot(emp.NumCompaniesWorked,bins=10)
sns.distplot(emp.TrainingTimesLastYear,bins=10)



####Bi Variate -Categorical predictor with Target
# sns.countplot(emp['TravelProfile'],hue=emp['Attrition'])
# sns.countplot(emp['Department'],hue=emp['Attrition'])
# sns.countplot(emp['EducationField'],hue=emp['Attrition'])
# sns.countplot(emp['Gender'],hue=emp['Attrition'])
# sns.countplot(emp['Designation'],hue=emp['Attrition'])
# sns.countplot(emp['MaritalStatus'],hue=emp['Attrition'])
# sns.countplot(emp['Involvement'],hue=emp['Attrition'])
# sns.countplot(emp['WorkLifeBalance'],hue=emp['Attrition'])
# sns.countplot(emp['JobSatisfaction'],hue=emp['Attrition'])
# sns.countplot(emp['OverTime'],hue=emp['Attrition'])
# sns.countplot(emp['ESOPs'],hue=emp['Attrition'])


####Duplicate analysis 
dups=emp.duplicated()
print("The num of duplicate records are =",dups.sum())


####outlier analysis
#emp.boxplot()

print('Shape before Outliers Treatment',emp.shape)

ll1,ul1=getupperlower_outlier(emp['Age'])
out1=emp.loc[(emp['Age']>ul1)|(emp['Age']<ll1),]
print(out1.shape)

ll2,ul2=getupperlower_outlier(emp['PercentSalaryIncrement'])
out2=emp.loc[(emp['PercentSalaryIncrement']>ul2)|(emp['PercentSalaryIncrement']<ll2),]
print(out2.shape)

#Office hours no outliers
ll3,ul3=getupperlower_outlier(emp['OfficeHours'])
out3=emp.loc[(emp['OfficeHours']>ul3)|(emp['OfficeHours']<ll3),]
print(out3.shape)


ll4,ul4=getupperlower_outlier(emp['YearsSinceLastPromotion'])
out4=emp.loc[(emp['YearsSinceLastPromotion']>ul4)|(emp['YearsSinceLastPromotion']<ll4),]
print(out4.shape)

ll5,ul5=getupperlower_outlier(emp['MonthlySalary'])
out5=emp.loc[(emp['MonthlySalary']>ul5)|(emp['MonthlySalary']<ll5),]
print(out5.shape)

ll6,ul6=getupperlower_outlier(emp['TotalWorkExp'])
out6=emp.loc[(emp['TotalWorkExp']>ul6)|(emp['TotalWorkExp']<ll6),]
print(out6.shape)

ll7,ul7=getupperlower_outlier(emp['YearsWithCurrManager'])
out7=emp.loc[(emp['YearsWithCurrManager']>ul7)|(emp['YearsWithCurrManager']<ll7),]
print(out7.shape)

ll8,ul8=getupperlower_outlier(emp['DistanceFromHome'])
out8=emp.loc[(emp['DistanceFromHome']>ul8)|(emp['DistanceFromHome']<ll8),]
print(out8.shape)


ll9,ul9=getupperlower_outlier(emp['NumCompaniesWorked'])
out9=emp.loc[(emp['NumCompaniesWorked']>ul9)|(emp['NumCompaniesWorked']<ll9),]
print(out9.shape)

#No oulliers with trainingtimeslastyear
ll10,ul10=getupperlower_outlier(emp['TrainingTimesLastYear'])
out10=emp.loc[(emp['TrainingTimesLastYear']>ul10)|(emp['TrainingTimesLastYear']<ll10),]
print(out10.shape)


##Recently_upskilled ,Remotework,StockOptionLevel,JobLevel,Yof Joining -Are categorical encoded so outliers dont matter here
####Missing and Bad data 
print(emp.isnull().sum())
emp.info()

catcols,numcols=get_num_cat_colnames(emp)


##Imputation ##
from sklearn.impute import KNNImputer
imputer=KNNImputer(
         n_neighbours=5,
         weights='distance',
         metric='nan_euclidean',#the metric to find the neighbours
        add_indicator=False #whether to add missing indicator  
    )






##Replacing missing values in categorical with MODE
for i in emp[['College_Tier','EnvironmentSatisfaction1','EnvironmentSatisfaction2','JobSatisfaction1','JobSatisfaction2','JobInvolvement1','JobInvolvement2','PerformanceRating1','PerformanceRating2']]:
    mode = emp[i].mode()
    print("Mode for this column is =",mode)
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
for k in emp[['YearsSinceLastPromotion','NumCompaniesWorked']]:
    median = emp[k].median()
    print("Median for this column is = ",median)
    print("No. Of NULL values in this column are",emp[k].isnull().sum())
    emp[k].fillna(median,inplace=True)
emp.isnull().sum()    

####Outlier treatment after imputation
#emp.boxplot()
#plt.xticks(rotation=90);
sns.boxplot(emp['Age'])
sns.boxplot(emp['PercentSalaryIncrement'])
sns.boxplot(emp['YearsSinceLastPromotion'])
sns.boxplot(emp['TotalWorkExp'])
sns.boxplot(emp['MonthlySalary'])
sns.boxplot(emp['YearsWithCurrManager'])
sns.boxplot(emp['DistanceFromHome'])
sns.boxplot(emp['NumCompaniesWorked'])



# for j in emp[['Age','PercentSalaryIncrement','YearsSinceLastPromotion','TotalWorkExp','MonthlySalary','YearsWithCurrManager','DistanceFromHome','NumCompaniesWorked']]:
#     emp[j]=treat_outlier_5_95(emp[j])  
#     print("Winsorization complete for column=",j)

##emp.boxplot()
##plt.xticks(rotation=90);

####Scaling -not reqd for CART and RF models

####Multicollinearity 
#sns.pairplot(emp)
#plt.show()
##Heatmap with masking 
plt.figure(figsize=(12,7))
sns.heatmap(emp.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(emp.corr()) )
plt.show()


####Encoding 

# #Converting object variables to corresponding categorical codes 
# for p in emp[['Region','BusinessTravel','MaritalStatus','Gender','Department','HighestEducation']]: 
#         emp[p] = pd.Categorical(emp[p]).codes 
#         print('Done for Column :',p)
        
##One Hot 
dummies=pd.get_dummies(emp[["Region", "BusinessTravel","MaritalStatus","Gender","Department","HighestEducation"]],
                       columns=["Region", "BusinessTravel","MaritalStatus","Gender","Department","HighestEducation"],
                       prefix=["REGION_", "BIZ_TRAVEL_","MARITAL_","GENDER_","DEPT_","EDUC_"],
                       drop_first=True)
col_to_rem=['Region', 'BusinessTravel','MaritalStatus','Gender','Department','HighestEducation']
emp = pd.concat([emp, dummies], axis=1)
emp.drop(col_to_rem,axis = 1,inplace=True)
print(emp.columns)
print(emp.shape)

##Encoding categorical nominal variables to numerical values as per order-label encoding
encobj_desig=OrdinalEncoder(categories=[['Tier-1' ,'Tier-2' ,'Tier-3']])
emp['College_Tier']=encobj_desig.fit_transform(emp[["College_Tier"]])    
emp['College_Tier'].value_counts().plot(kind='bar')    

emp.to_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\real hackathon\emp_onehot.csv')

####demerging
test_unknown=emp.loc[(emp['Ind']=="unknown_test")]
print(test_unknown.shape)
print(test_unknown['Employeeturnover'].unique())
test_unknown.drop(['Employeeturnover','Ind'],axis=1,inplace=True)
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



####train and validate split
all_labels=emp_train['Employeeturnover']
all_ind_ds=emp_train.drop(["Employeeturnover"],axis=1)
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
     
# Scaling reqd for neuralnet 
# sobj=StandardScaler()
# data_scaled_trainsetnn=sobj.fit_transform(trainsetnn)
# data_scaled_testsetnn=sobj.fit_transform(testsetnn)
# data_scaled_test_unknown=sobj.fit_transform(test_unknown)
####Preparing trainset for xgboost

start_time1=time.monotonic()
print(start_time1)
from xgboost import XGBClassifier
# bst_xgb_mod1=xgb.train(params=param,
#                        dtrain=dtrain,
#                        num_boost_round=999,
#                        early_stopping_rounds=10,
#                        evals=(dval,"validationset"),
#                        verbose_eval=True)




xgbobj=XGBClassifier(random_state=123,objective="binary:logistic")
eval_set=[(testset,test_labels)]

param_grid = {
     
    'n_estimators': [700],
    'max_depth': [4],
    #'min_samples_leaf': [1,2,3],
    #'min_samples_split': [2,3],
    'eta':[0.1],
    'early_stopping_rounds':[10],
    'subsample':[0.8],
    'ColSamplebyTree':[0.6],
    #'Lambda':[0.5],
    #'alpha':[0.2]
}


xgb_mod=GridSearchCV(estimator=xgbobj,param_grid=param_grid,cv=10,verbose=20,scoring='f1')
xgb_mod.fit(trainset,train_labels)
##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
train_time=timedelta(seconds=end_time1 - start_time1)
print("training time of xgb model= ",train_time)
print("Best params are =",xgb_mod.best_params_)
print("Best score after gridsearch =",xgb_mod.best_score_)
xgb_mod_bst=xgb_mod.best_estimator_

# ##make predictions

pred_train=xgb_mod_bst.predict(trainset)
print(pred_train.size)
print("Accuracy on TRAIN set =",f1_score(pred_train,train_labels))
acc_train_xgb=f1_score(pred_train,train_labels)

pred_val=xgb_mod_bst.predict(testset)
print(pred_val.size)
print("Accuracy on VALIDATE set= ",f1_score(pred_val,test_labels))
acc_test_xgb=f1_score(pred_val,test_labels)

print("Actual Accuracy on VALIDATE set= ",accuracy_score(pred_val,test_labels))
##AUC score Trainset 
probs5 = xgb_mod_bst.predict_proba(trainset)
# keep probabilities for the positive outcome only
probs5 = probs5[:, 1]
# calculate AUC
auc5 = roc_auc_score(train_labels, probs5)
print('AUC on Trainset for XGB model = %.3f' % auc5)
##Calc ROC curve and threshold 
fpr5, tpr5, thresholds5 = roc_curve(train_labels, probs5)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr5, tpr5, marker='.')
# show the plot
plt.show()

##AUC score Test
probs6 = xgb_mod_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs6= probs6[:, 1]
# calculate AUC
auc6 = roc_auc_score(test_labels, probs6)
print('AUC of XGB model on TEST SET  %.3f ' % auc6)
##Calc ROC curve and threshold 
fpr6, tpr6, thresholds6 = roc_curve(test_labels, probs6)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr6, tpr6, marker='.')
# show the plot
plt.show()


xgb_mod2=xgb_mod_bst.fit(all_ind_ds,all_labels)
pred2=xgb_mod2.predict(all_ind_ds)
print(pred2.size)
print("f1 accuracy on best model BUT fitted to all training data =",f1_score(pred2,all_labels))
acc_full_xgb=f1_score(pred2,all_labels)
auc7 = roc_auc_score(all_labels, pred2)
print('AUC (all data ) of XGB best model BUT fitted to all training data   %.3f ' % auc7)

# pred_whole=xgb_mod_bst_bst.predict(all_ind_ds)
# print(pred_whole.size)
# print("Accuracy on WHOLE TRAIN set= ",accuracy_score(pred_whole,all_labels))

# ##Save the trained model on disk 
# pickle.dump(xgb_mod_bst,open("xgbmod1_withpickle.dat","wb"))
# print("saved model to :xgbmod1_withpickle.dat")

#loaded_xgb_mod_bst_disk=pickle.load(open("xgbmod1_temp4_ulll.dat","rb"))
#print("Loaded from disk : loaded_xgb_mod_bst_disk-from file xgbmod1_temp4_ulll.dat ")

##Importance 
plot_importance(xgb_mod_bst)



# ##optimal threshold 

# for j in np.arange(0.1,1,0.1):
#     custom_prob = j #defining the cut-off value of our choice
#     custom_cutoff_data=[]#defining an empty list
#     for i in range(0,len(train_labelsnn)):#defining a loop for the length of the train data
#         if np.array(probs4)[i] > custom_prob:#issuing a condition for our probability values to be 
#             #greater than the custom cutoff value
#             a=1#if the probability values are greater than the custom cutoff then the value should be 1
#         else:
#             a=0#if the probability values are less than the custom cutoff then the value should be 0
#         custom_cutoff_data.append(a)#adding either 1 or 0 based on the condition to the end of the list defined by us
#     print(round(custom_prob,3),round(metrics.f1_score(train_labelsnn,custom_cutoff_data),4))
# ##0.6-0.9986 highest



# pred_test_nn=nn_mod1_bst.predict(data_scaled_testsetnn)

# print(confusion_matrix(test_labelsnn,pred_test_nn))
# acc_nn_test=f1_score(test_labelsnn,pred_test_nn)
# print(acc_nn_test)
# print("F1 Score on the VALIDATE set  =",f1_score(test_labelsnn,pred_test_nn))
# probs3 = nn_mod1_bst.predict_proba(data_scaled_testsetnn)
# # keep probabilities for the positive outcome only
# probs3 = probs3[:, 1]
# # calculate AUC
# auc3 = roc_auc_score(test_labelsnn, probs3)
# print('AUC on VALIDATE set: %.3f' % auc3)


# data_pred_custom_cutoff_validate=[]
# for i in range(0,len(probs3)):
#     if np.array(probs3)[i]>0.6:
#         a=1
#     else:
#         a=0
#     data_pred_custom_cutoff_validate.append(a)

####Result collation -Accuracy on Validate set 
index=['F1 Score ', 'AUC ']
dataf = pd.DataFrame({
        'XGB Train':[acc_train_xgb,auc5],
        'XGB Test':[acc_test_xgb,auc6],
        'XBGFull':[acc_full_xgb,auc7]},index=index)
round(dataf,5)

####Testset processing 
print(unktestset_orig.shape)
id=unktestset['ID']
print(test_unknown.shape)
#print(len(data_scaled_test_unknown))

res_cols=['ID','Employeeturnover']


#pred_unk_cart=cart_mod1_bst.predict(test_unknown)
#pred_unk_cart=pd.Series(pred_unk_cart.astype(int),name="Employeeturnover")

pred_unk_xgb=xgb_mod_bst.predict(test_unknown)
pred_unk_xgbfull=xgb_mod2.predict(test_unknown)


# probs_nn_unk = nn_mod1_bst.predict_proba(data_scaled_test_unknown)
# probs_nn_unk = probs_nn_unk[:, 1]
# data_pred_custom_cutoff_unk=[]
# for i in range(0,len(probs_nn_unk)):
#     if np.array(probs_nn_unk)[i]>0.6:
#         a=1
#     else:
#         a=0
#     data_pred_custom_cutoff_unk.append(a)

# pred_unk_nn=pd.Series(data_pred_custom_cutoff_unk.astype(int),name="Attrition")
pred_unk_xgb=pd.Series(pred_unk_xgb.astype(int),name="Employeeturnover")
pred_unk_xgbfull=pd.Series(pred_unk_xgbfull.astype(int),name="Employeeturnover")
##combine=np.vstack((id,pred_unk_cart)).T
#combine=np.vstack((id,pred_unk_cart)).T
#result_cart=pd.concat([id,pred_unk_cart],axis=1)
result_xgb=pd.concat([id,pred_unk_xgb],axis=1)
result_xgbfull=pd.concat([id,pred_unk_xgbfull],axis=1)

####Final submission file

##Dynamic file name with datetime stamp
snapshotdate = datetime.datetime.today().strftime('%d-%m-%Y_%H_%M_%S')
print(snapshotdate)
myfile_name='xgb_'+snapshotdate+'.csv'
myfile_name1='xgbfull_'+snapshotdate+'.csv'
print(myfile_name)
p='E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\real hackathon\\submission files\\'
print("printing output file for model trained on trainset and fullset at specified location ")
result_xgb.to_csv(p+myfile_name,index=False)
result_xgbfull.to_csv(p+myfile_name1,index=False)
