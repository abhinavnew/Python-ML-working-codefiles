# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:56:13 2021

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
from scipy.stats import chi2_contingency
from scipy.stats import   ttest_1samp,ttest_ind
from scipy.stats import variation  
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from sklearn.preprocessing import StandardScaler
sns.set(color_codes=True)

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
##correlated columns,same value columns


DF = pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week 7-ANOVA\\paul-newfood.csv')

##converting to categorical data type

DF.PriceLevel=pd.Categorical(DF.PriceLevel)
DF.AdLevel=pd.Categorical(DF.AdLevel)

form='Sales ~ C(PriceLevel)+ C(AdLevel)'
mod=ols(form,DF).fit()

ano_tab=anova_lm(mod)

print(ano_tab)



car = pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week 7-ANOVA\\car_insurance_rate.csv',delimiter='\t')

car_melt=car.melt(var_name='city')

##Total num of observations 
n=car.shape[0]*car.shape[1]
##Total num of groups 
k=car.shape[1]

##Deg of freedom between groups 
dfbgroups=k-1
##Deg of freedom within groups
dfwgroups=n-1

##Executing ANOVA 
form2='value ~ city'
mod2=ols(form2,car_melt).fit()
ano_tab2=anova_lm(mod2)
print(ano_tab2)


fastfood=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week 7-ANOVA\\FastFood-1 (1).csv')

##Total num of observations
n=24
##Total num of groups -unique type
k=4

##Deg of freedom between groups 
dfb_grps=k-1
##Deg of freedom within groups 
dfw_grps=n-1

##Executing ANOVA 
form3='Sales ~ C(FastFoodNames)'
mod3=ols(form3,fastfood).fit()
ano_tab3=anova_lm(mod3)
print(ano_tab3)

##P val 0.16 >0.05 hence fail to reject Null hyp :categories have no effect on sales

##Exploratory analysis 

##replacing object Null values like ? with NaN

df['col1'].replace('?',np.NaN,inplace=True)
df['col1']=df['col1'].astype('float64')
##check and replace with col3 if condition true else let it be col2
df['col2']=np.where(df['col2']==-1,df['col3'],df['col2'])

##impute with mean when bad val -1
df['col5'].replace(-1,df['col5'].mean(),inplace=True)

##impute with mean/median for missing values (if no outliers then safe)
df.col6.fillna(df.col6.mean(),inplace=True)

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df_numericcols=pd.DataFrame(imputer.fit_transform(df_numericcols),columns=df_numericcols.columns)


##to display values in percentge ,0-1 decimals put normalize 
df['col7'].value_counts(normalize=True)


referaldata=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week8 EDA\\Referral_Join_Prediction.csv')

cat=[]
num=[]
for i in referaldata.columns :
    if referaldata[i].dtype=="object":
        cat.append(i)
    else :
        num.append(i)
print(cat)
print(num)        


##scaling 


scal=StandardScaler().fit(referaldata[num])
data_stand=scal.transform(referaldata[num])
data_stand=pd.DataFrame(data_stand,columns=referaldata[num].columns)


        
credit=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week8 EDA\\credit_card (1).csv')



slump=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week8 EDA\\Slump Test (1).csv')

##check for duplicates 
dups=slump.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
slump[dups]

##missing values 
slump.info()
slump.isnull().sum()


##Outliers
df=slump.copy()
ax=df.boxplot(figsize=(30,20))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
plt.tight_layout()
plt.show()

##SP ;SLUMP;Strength have outliers

##List out outlier rows 


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of ths column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


lr,ur=remove_outlier(slump['Fine'])
print("lower range",lr, "and upper range", ur)

slump.loc[(slump['SP']>16.0)]


##DOM SMET 
##replace outlier by winsorization 
df['referral_current_salary1']=np.where(df['referral_current_salary1']>ur,ur,df['referral_current_salary1'])
df['referral_current_salary1']=np.where(df['referral_current_salary1']<lr,lr,df['referral_current_salary1'])

# Method III Min-Max method
from sklearn.preprocessing import MinMaxScaler
# build the scaler model
scaler = MinMaxScaler().fit(df[num])
# transform the test test
data_minmax = scaler.transform(df[num])
data_minmax=pd.DataFrame(data_minmax, columns=df[num].columns)
data_minmax.describe()


##
co_rel=slump.corr()
round(co_rel,5)


df=slump.copy()
df.boxplot(figsize=(4,4),grid=True, rot=90);

df.boxplot(figsize=(1,4),grid=True);

df.boxplot(figsize=(15,4));

df.boxplot(figsize=(15,4),grid=False);


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(slump.Slag, kde=False, ax=axs[0]) 
sns.boxplot(x= 'BILL_AMT_AUG', data=credit, ax=axs[1]) 

slump['Slag'].plot.hist(density=True)



