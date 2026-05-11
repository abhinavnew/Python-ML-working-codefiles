# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:58:16 2021

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


def treat_outlier(x):
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
##PCA exercise for redwine
redwine=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week 9-PCA\\Redwine+Quality.csv')

df=redwine.copy()
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df.shape)


##find duplicates AND drop them 
dups=df.duplicated()
print("no of duplciates in this data set ",dups.sum())
df[dups]
df.drop_duplicates(inplace=True)

##outliers


df.boxplot();
plt.xticks(rotation = 45);



# ##display outlier in a partcular column
# ##Get list of categorical and numerical columns 
# hl_cat=[]
# hl_num=[]
# for i in homeloan.columns :
#     if homeloan[i].dtype=="object":
#         hl_cat.append(i)
#     else :
#         hl_num.append(i)
# print(hl_cat)
# print(hl_num)     

##winsorization -replace outliers with upper and lower limits calc 
##list_num = ['INCOME', 'TRAVEL TIME', 'MILES CLOCKED']
# for i in hl_num:
#     LL, UL = remove_outlier(homeloan[i])
#     homeloan[i] = np.where(homeloan[i] > UL, UL, homeloan[i])
#     homeloan[i] = np.where(homeloan[i] < LL, LL, homeloan[i])

##Outlier treatment with winsorization to 5th and 95th percentile
for i in df.columns:    
    df[i]=treat_outlier(df[i])

##checking boxplot after outlier treatment

df.boxplot();
plt.xticks(rotation = 45);

##Outliers removed 

##Missing values --None found

df.isnull().sum()


##scaling 

scal=StandardScaler().fit(df)
data_stand=scal.transform(df)
data_stand=pd.DataFrame(data_stand,columns=df.columns)

##multicollinearity

corrmat=df.corr()
corrmat[np.abs(corrmat)<.40] = 0
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
    ax2.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat_homeloan.csv")

##Step 1-Calc covariance matrix 
cov_matrix=np.cov(data_stand.T)

# Step 2- Get eigen values and eigen vector
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('\n Eigen Values \n %s', eig_vals)
print('\n')
print('Eigen Vectors \n %s', eig_vecs)

tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)


# Step 3 View Scree Plot to identify the number of components to be built
plt.figure(figsize=(12,7))
sns.lineplot(y=var_exp,x=range(1,len(var_exp)+1),marker='o')
plt.xlabel('Number of Components',fontsize=15)
plt.ylabel('Variance Explained',fontsize=15)
plt.title('Scree Plot',fontsize=15)
plt.grid()
plt.show()



# Step 4 Apply PCA for the number of decided components to get the loadings and component output

# Using scikit learn PCA here. It does all the above steps and maps data to PCA dimensions in one shot
from sklearn.decomposition import PCA
# NOTE - we are generating only 8 PCA dimensions (dimensionality reduction from 33 to 8)
pca = PCA(n_components=4, random_state=123)
df_pca = pca.fit_transform(data_stand)
df_pca.transpose() # Component output


##################################################

fb_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\week 9-PCA\\FB-1.csv')

fb=fb_orig.copy()
##find duplicates AND drop them 
dups2=fb.duplicated()
print("The num of duplicates ",dups2.sum())
fb[dups2]
fb.drop_duplicates(inplace=True)

##outliers

fb.boxplot();
plt.xticks(rotation = 45);
##Outlier treatment with winsorization to 5th and 95th percentile
# ##Get list of categorical and numerical columns 
fb_cat=[]
fb_num=[]
for i in fb.columns :
    if fb[i].dtype=="object":
        fb_cat.append(i)
    else :
        fb_num.append(i)
print(fb_cat)
print(fb_num) 

for i in fb_num:    
    fb[i]=treat_outlier(fb[i])

##checking boxplot after outlier treatment

fb.boxplot();
plt.xticks(rotation = 90);
##Outliers removed 

##Missing values --None found

fb.isnull().sum()

##scaling 
fb.drop(['status_id'],axis=1,inplace=True)
scal=StandardScaler().fit(fb)
data_stand=scal.transform(fb)
data_stand=pd.DataFrame(data_stand,columns=fb.columns)

##multicollinearity

corrmat=data_stand.corr()
corrmat[np.abs(corrmat)<.40] = 0
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
    ax2.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat_fbdata.csv")


##Step 1-Calc covariance matrix 
cov_matrix=np.cov(data_stand.T)

# Step 2- Get eigen values and eigen vector
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print('\n Eigen Values \n %s', eig_vals)
print('\n')
print('Eigen Vectors \n %s', eig_vecs)

tot = sum(eig_vals)
var_exp = [( i /tot ) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cumulative Variance Explained", cum_var_exp)


# Step 3 View Scree Plot to identify the number of components to be built
plt.figure(figsize=(12,7))
sns.lineplot(y=var_exp,x=range(1,len(var_exp)+1),marker='o')
plt.xlabel('Number of Components',fontsize=15)
plt.ylabel('Variance Explained',fontsize=15)
plt.title('Scree Plot',fontsize=15)
plt.grid()
plt.show()



# Step 4 Apply PCA for the number of decided components to get the loadings and component output

# Using scikit learn PCA here. It does all the above steps and maps data to PCA dimensions in one shot
from sklearn.decomposition import PCA
# NOTE - we are generating only 8 PCA dimensions (dimensionality reduction from 33 to 8)
pca = PCA(n_components=4, random_state=123)
df_pca = pca.fit_transform(data_stand)
df_pca.transpose() # Component output


