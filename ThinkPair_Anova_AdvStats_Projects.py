# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:42:00 2021

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

stock = pd.read_excel('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\STOCK-Q2 (1).xlsx')

##Total num of observations 
n=stock.shape[0]
##Total num of groups 
k=stock['Sector'].nunique()

##Deg of freedom between groups 
dfbgroups=k-1
##Deg of freedom within groups
dfwgroups=n-1

##Executing ANOVA 
form_2='Stock_Return ~ C(Sector)'
mod2=ols(form_2,stock).fit()
ano_tab2=anova_lm(mod2)
print(ano_tab2)

##p value 0.081 > 0.05 significance level ,hence fail to reject NULL hyp ie all sample means are equal ,ie returns from all sectors are equal
##Sector has no impact on stock returns 

##To verify this claim we can run independent t test with 2 sector groups each
##Groups are (Consumer ,service ),(Consumer,Industrial),(Service Industrial)

consumer_samp=stock.loc[(stock['Sector']=='Consumer'),'Stock_Return']
industrial_samp=stock.loc[(stock['Sector']=='Industrial'),'Stock_Return']
service_samp=stock.loc[(stock['Sector']=='Service'),'Stock_Return']

##Step 1: Define null and alternative hypotheses
##H0:Sample A mean EQUAL to Sample B mean 
##HA:Sample A Mean NOT EQUAL to Sample B Mean 

##Step 2: Decide the significance level
##Here we select $\alpha$ = 0.05 and the population standard deviation is not known.

##Step 3: Identify the test statistic
##* We have two samples and we do not know the population standard deviation.
##* Sample sizes for both samples are  same.All pairs ie (A,B)(B,C)(A,C)
##* The sample is not a large sample, n < 30. So you use the t distribution and the $t_{STAT}$ test statistic for two sample unpaired test.

##Step 4: Calculate the p - value and test statistic

t_statistic1, p_value1  = ttest_ind(consumer_samp,industrial_samp,equal_var=False)
print('tstat',t_statistic1)    
print('P Value',p_value1)  

## ttest 1 Fails p value < 0.05 H0 is rejected this is a contradiction

t_statistic2, p_value2  = ttest_ind(consumer_samp,service_samp,equal_var=False)
print('tstat',t_statistic2)    
print('P Value',p_value2)  

t_statistic3, p_value3  = ttest_ind(service_samp,industrial_samp,equal_var=False)
print('tstat',t_statistic3)    
print('P Value',p_value3)  

##pval > 0.05 hence Null hypothesis is not rejected (accepted )ie sample A mean equal to sample B mean 







