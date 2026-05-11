# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:17:23 2021

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
import math as math
import scipy.stats as stats
import datetime 
from scipy.stats import chi2_contingency
from scipy.stats import   ttest_1samp,ttest_ind
from scipy.stats import variation  
import statistics as sts
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from sklearn.preprocessing import StandardScaler
sns.set(color_codes=True)

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


cold_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\smdm re do\\Cold_Storage_Temp_Data_ (1).csv')

cold=cold_orig.copy()

a=cold.groupby(['Season'],as_index=False)['Temperature '].mean()
print(a)

cold['Season'].value_counts()

b=cold['Temperature '].mean()
print(b)

##Standard deviation for full year 
c=cold['Temperature '].std()
print(c)

z_2=(2-b)/c
z_4=(4-b)/c

##Prob of falling below 2 C
p_below2=stats.norm.cdf(z_2)

##Prob of falling above 4C
p_above4=1-stats.norm.cdf(z_4)

##Prob of falling below 2 and above 4 
print(p_below2+p_above4)

1-(stats.norm.cdf(z_4)-stats.norm.cdf(z_2))

##total probablity of temp falling out of range =3.1% ;hence fine =10% of AMC


##Prob 2

cold_mar_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\smdm re do\\Cold_Storage_Mar2018_ (1).csv')

cold_mar=cold_mar_orig.copy()

temp_only=cold_mar[' Temperature ']


##Step 1 Hypothesis formation 
##H0-mean temperature <=3.9C
##HA-mean temperature >3.9C

##Step 2 Alpha=0.01
alpha_level = 0.01

##Step 3 Identify the statistic  and test -z test as the sample size >30 


##Step 4: Calculate the p - value and test statistic
n_samplesize=35
meu=3.9
x_bar=temp_only.mean()
sig_bar=sts.stdev(temp_only)
intr=((x_bar-meu)/sig_bar)
t_st=intr*math.sqrt(n_samplesize)
t_st2=intr
print(intr)
print(x_bar)
print(sig_bar)
print(t_st)
print(t_st2)
p_val=stats.norm.cdf(t_st)
p_val2=stats.norm.cdf(t_st2)
print(p_val)
print(p_val2)

##Step 5 compare and confirm or reject hypothesis 
if p_val < alpha_level:
    print('H0 REJECTED : We have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
   
else:
    print('H0 ACCEPTED: We do not have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
   
if p_val2 < alpha_level:
    print('H0 REJECTED : We have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
   
else:
    print('H0 ACCEPTED :We do not have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
       
   





