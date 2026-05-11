# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:04:39 2021

@author: Abhinav.Bajpai
"""

import gc
##Clear variable/objects from workspace to free up memory
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
dir()

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range

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




## using a custom defined function decalared earlier 

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




df.boxplot(figsize=(15,4));


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(slump.Slag, kde=False, ax=axs[0]) 
sns.boxplot(x= 'BILL_AMT_AUG', data=credit, ax=axs[1]) 

slump['Slag'].plot.hist(density=True)

##PCA exercise for homeloan across various branches of bank (INDIA)
homeloan=pd.read_excel('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week 9-PCA\\Short Term Home Loans.xlsx')

##duplicates 

df=homeloan.copy()

dups=df.duplicated()
print("no of duplciates in this data set ",dups.sum())
df[dups]

##outliers

ax5=sns.barplot(df)
ax5.set_xticklabels(
    ax5.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')


##display outlier in a partcular column
##Get list of categorical and numerical columns 
hl_cat=[]
hl_num=[]
for i in homeloan.columns :
    if homeloan[i].dtype=="object":
        hl_cat.append(i)
    else :
        hl_num.append(i)
print(hl_cat)
print(hl_num)     

##winsorization -replace outliers with upper and lower limits calc 
##list_num = ['INCOME', 'TRAVEL TIME', 'MILES CLOCKED']
for i in hl_num:
    LL, UL = remove_outlier(homeloan[i])
    homeloan[i] = np.where(homeloan[i] > UL, UL, homeloan[i])
    homeloan[i] = np.where(homeloan[i] < LL, LL, homeloan[i])


ax6=sns.barplot(homeloan)
ax6.set_xticklabels(
    ax6.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

sns.barplot(df)


##lr,ur=remove_outlier(df['gross_income'])
##print("lower range",lr, "and upper range", ur)
##df.loc[(df['gross_income']>ur)]

##Missing values 
homeloan.isnull().sum()

##scaling 
sns.pairplot(homeloan ,diag_kind='kde')
scal=StandardScaler().fit(homeloan[hl_num])
data_stand=scal.transform(homeloan[hl_num])
data_stand=pd.DataFrame(data_stand,columns=homeloan[hl_num].columns)

##multicollinearity


corrmat=homeloan.corr()
corrmat[np.abs(corrmat)<.70] = 0
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
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat_homeloan.csv")






