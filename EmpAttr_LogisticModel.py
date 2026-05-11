# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:48:26 2020

@author: Abhinav.Bajpai
"""

##Item listing in a Store's branches 



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
import datetime
import sklearn
import xgboost as xgb

from sklearn.model_selection import train_test_split

##Read dataset 
emp_orig=pd.read_csv("E:\AbhinavB\Kaggle\IBM HR Analytics Employee Attrition\ibm-hr-analytics-employee-attrition-performance\WA_Fn-UseC_-HR-Employee-Attrition.csv")


emp_orig.head(100)
starttime=datetime.datetime.now()
print(starttime)
print(emp_orig.shape)
list(emp_orig.columns)
print(emp_orig.dtypes)
emp_orig.head(10)

mydf_temp=emp_orig.copy()
print(mydf_temp.shape)
a=mydf_temp['BusinessTravel'].unique()
print(a)

##label encoding -->convert to category and then to numeric

objcoldf=mydf_temp.select_dtypes(include=['object'].copy())
objcoldf.head(10)
 
print(objcoldf.dtypes)
objcoldf=objcoldf.astype('category')
print(objcoldf.dtypes)
print(objcoldf.head(100))
objcoldf['Attrition']=objcoldf['Attrition'].cat.codes
objcoldf['BusinessTravel']=objcoldf['BusinessTravel'].cat.codes
objcoldf['Department']=objcoldf['Department'].cat.codes
objcoldf['EducationField']=objcoldf['EducationField'].cat.codes
objcoldf['Gender']=objcoldf['Gender'].cat.codes
objcoldf['JobRole']=objcoldf['JobRole'].cat.codes
objcoldf['MaritalStatus']=objcoldf['MaritalStatus'].cat.codes
objcoldf['Over18']=objcoldf['Over18'].cat.codes
objcoldf['OverTime']=objcoldf['OverTime'].cat.codes
##adding suffix to column names 
objcoldf=objcoldf.add_suffix('_numeric')
objcoldf.dtypes
list(objcoldf.columns)

##Merging with old data and keeping only numeric columns
print(mydf_temp.head(10))
print(mydf_temp.shape)
print(mydf_temp.dtypes)
objcoldf=objcoldf.reset_index(drop=True)
totdf=pd.concat([mydf_temp,objcoldf],axis=1)
type(totdf)
totdf.head(10)
print(totdf.shape)
print(totdf.dtypes)
##totdf.drop(totdf.columns[[1,2,4,7,11,15,17,21,22]],axis=1,inplace=True)
totdf=totdf.select_dtypes(exclude=['object'])
print(totdf.shape)
totdf.head(10)

totdf.to_csv(r'E:\python_codefiles\totdf.csv',index=False)

##Splitting the dataset into (Train+validate)and Test sets

bigtrain,test=train_test_split(totdf,test_size=0.4)

print(bigtrain.shape,test.shape)

train,validate=train_test_split(bigtrain,test_size=0.3)
print(train.shape,validate.shape)
train.head(10)

vec1 =np.arange(1,10)
len(vec1)
print(vec1)

vec2=np.arange(11,20)
len(vec2)
print(vec2)

print(vec1+vec2)
a=pd.dataframe
print(vec1*vec2)

mat1=np.array([vec1,vec2])
print(mat1)
print(mat1(1,1))

list5=[4.99,5,900]
labels=['No1','No2','No3']
series3=pd.Series(data=list5,index=labels)
print(series3)


##Visuals using seaborn and matplotlib

##Travel frequently Vs Count but attrition wise side by side graph






