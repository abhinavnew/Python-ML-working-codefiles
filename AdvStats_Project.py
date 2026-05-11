# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:40:26 2021

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

educ_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\Education+-+Post+12th+Standard.csv')

educ=educ_orig.copy()
##Duplicates 
dups3=educ.duplicated()
print("The no of duplicates in this data set are ",dups3.sum())
educ[dups3]
educ.drop_duplicates(inplace=True)
educ.shape

##Outliers

educ.boxplot();
plt.xticks(rotation = 90);

##Lots of outliers are seen ,these need to be treated 

##getting all numerical and categorical column name list 


educ_cat=[]
educ_num=[]

for i in educ.columns:
      if educ[i].dtype=="object":
          educ_cat.append(i)
      else:
          educ_num.append(i)
print(educ_cat)     
print(educ_num)     

##winsorization -replace outliers with upper and lower limits calc 
##list_num = ['INCOME', 'TRAVEL TIME', 'MILES CLOCKED']
# for i in hl_num:
#     LL, UL = remove_outlier(homeloan[i])
#     homeloan[i] = np.where(homeloan[i] > UL, UL, homeloan[i])
#     homeloan[i] = np.where(homeloan[i] < LL, LL, homeloan[i])

##Outlier treatment with winsorization to 5th and 95th percentile
for i in educ_num:    
    educ[i]=treat_outlier_5_95(educ[i])

##checking boxplot after outlier treatment

educ.boxplot();
plt.xticks(rotation = 90);

##Now most outliers are treated 

##Missing values 
##no outliers then mean impute else median ,categorical -mode impute 

educ.isnull().sum()[educ.isnull().sum()>0]
##No missing values found 


##Univariate analysis
def univariateAnalysis_numeric(column,nbins):
    print("Description of " + column)
    print("----------------------------------------------------------------------------")
    print(df[column].describe(),end=' ')
    print(" ")
    print("Coeff of Variation " + column)
    print("----------------------------------------------------------------------------")
    cv=(df[column].std()/df[column].mean())
    print(round(cv,2))
    plt.figure()
    print(" ")
    print("Distribution of " + column)
    print("----------------------------------------------------------------------------")
    sns.distplot(df[column], kde=False, color='g');
    plt.show()    
    plt.figure()
    print(" ")
    print("BoxPlot of " + column)
    print("----------------------------------------------------------------------------")
    ax = sns.boxplot(x=df[column])
    plt.show()
    
    

df=educ.copy()

univariateAnalysis_numeric('Apps',20)

for x in educ_num:
    univariateAnalysis_numeric(x,20)

##Apps are left skewed range between 0-2000 ,on avg every college recieves ~3000 apps
##Accepted apps 

##BiVariate analysis 
sns.pairplot(educ)

##multicollinearity

corrmat=educ.corr()
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
    rotation=90,
    horizontalalignment='right'
);
corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat_educdataset.csv")

##Analysis shows several parameters with high correlation like:
##Apps ~ Accept/Enroll ;Top10per ~Top25Perc ;F.Undergrad ~ Enroll ;phd ~ Terminal ;;
##Negatively correlated SF ratio ~ -(Outstate,Expend)

##Scaling is required as diff parameters on diff scale Top10perc in % terms ;App,enroll,accept are no of apps ;RoomBoard/Books are dollar amounts 

##scaling 

educ_num_df = educ.select_dtypes(include = ['float64', 'int64'])

scal=StandardScaler().fit(educ_num_df)
data_stand=scal.transform(educ_num_df)
data_stand_educ=pd.DataFrame(data_stand,columns=educ_num_df.columns)

# from scipy.stats import zscore
# df_num_scaled=df_num.apply(zscore)
# df_num_scaled.head()

data_stand_educ.boxplot(figsize=(20,3))
plt.xticks(rotation=90)
plt.show()

##PCA 
##Step 1-Calc covariance matrix 
cov_matrix=np.cov(data_stand_educ.T)

# Step 2- Get eigen values and eigen vector and printing both
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
eig_vals_r=eig_vals.round(decimals=2)
eig_vecs_r=eig_vecs.round(decimals=2)
print('\n Eigen Values \n %s', eig_vals_r)
print('\n')
print('Eigen Vectors \n %s', eig_vecs_r)


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

##With 3 PCs 68% of variance is explained 
##Eigen vectors indicate the direction of the PCs

# Step 4 Apply PCA for the number of decided components to get the loadings and component output

# Using scikit learn PCA here. It does all the above steps and maps data to PCA dimensions in one shot
from sklearn.decomposition import PCA
# NOTE - we are generating only 8 PCA dimensions (dimensionality reduction from 33 to 8)
pca = PCA(n_components=4, random_state=123)
df_pca = pca.fit_transform(data_stand_educ)
p=df_pca.transpose() # Component output

################################ANOVA 
##Problem 1 (a)

salary_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\SalaryData.csv')

salary=salary_orig.copy()


salary.Education=pd.Categorical(salary.Education)
salary.Occupation=pd.Categorical(salary.Occupation)

# ##Total num of observations 
# n=stock.shape[0]
# ##Total num of groups 
# k=stock['Sector'].nunique()

# ##Deg of freedom between groups 
# dfbgroups=k-1
# ##Deg of freedom within groups
# dfwgroups=n-1

salary['Education'].value_counts()

salary.groupby(['Education'],as_index=False).mean()

##Shapiro wilk test for normality 
#########################################################
##H0 :Salary follows a normal distribution 
##HA :Salary doesnot follow a normal distribution 
from scipy import stats
w,p_val=stats.shapiro(salary['Salary'])
print ("P value for shapiro test of normality= {}".format(p_val))
##p val <0.05 hence reject the null hypothesis 
##Distribution is not normal 


sns.boxplot(x='Education',y='Salary',data=salary,hue='Education')


# # ##Next, we need to test the assumption that at all three
# # levels of the factor fuel_type, population variance is 
# # equal. In other words, the homogeneity of 
# # variance assumption is satisfied. We may formulate the problem as:
# ##Test for homogeneity of variance 
#########################################################
# #H0 : all variances are equal across pops
# #HA: atleast one variance is different from the rest 
# statistic,p_value = stats.levene(salary['Salary'][salary['Education']=="Bachelors"], salary['Salary'][salary['Education']=="HS-Grad"], salary['Salary'][salary['Education']=="Doctrate"])    

bin_edges = np.arange(50000, 260200, 20) 
plt.hist(salary.Salary, 
         bins=bin_edges, density=False, 
         histtype='bar', color='b', 
         edgecolor='k', alpha=0.5);




##Executing 1 way ANOVA ::salary only dependent on education 

form1='Salary ~ C(Education)'
mod1=ols(form1,salary).fit()
ano_tab1=anova_lm(mod1)
print(ano_tab1)

##p value < 0.05 significance level ,hence fail to reject NULL hyp ie all sample means are equal ,ie returns from all sectors are equal
##Sector has no impact on stock returns 

sns.pointplot(x='Education', y='Salary', data=salary, hue='Occupation')

##1 way ANOVA for occupation 

salary['Occupation'].value_counts()
salary.groupby(['Occupation'],as_index=False).mean()
a4_dims = (7,7) 
fig, ax = plt.subplots(figsize=a4_dims)
a=sns.boxplot(x='Occupation',y='Salary',data=salary,hue='Occupation')
a.set_title("Salary w.r.t. Occupation (4 levels)",fontsize=15)
plt.show()
##Executing 1 way ANOVA ::salary only dependent on Occupation 

form2='Salary ~ C(Occupation)'
mod2=ols(form2,salary).fit()
ano_tab2=anova_lm(mod2)
print(ano_tab2)
##Interaction between the variables 



sns.pointplot(x='Education',y='Salary',data=salary,hue='Occupation')



form3='Salary ~ C(Education)+C(Occupation)+C(Education):C(Occupation)'
mod3=ols(form3,salary).fit()
ano_tab3=anova_lm(mod3)
print(ano_tab3)
