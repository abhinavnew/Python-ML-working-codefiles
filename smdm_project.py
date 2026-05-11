# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:25:42 2021

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

wholesale=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\Week6 -hypothesis\\Project work SMDM\\Wholesale+Customers+Data.csv")

wholesale['Tot_spend']=wholesale['Fresh']+wholesale['Milk']+wholesale['Grocery']+wholesale['Frozen']+wholesale['Detergents_Paper']+wholesale['Delicatessen']

##1.1 Use methods of descriptive statistics to summarize data. Which Region and which Channel seems to spend more? Which Region and which Channel seems to spend less?

wholesale.groupby(['Region','Channel'],as_index=False)['Tot_spend'].sum()

##Comparison on the basis of Region/Channel only 

wholesale.groupby(['Region'],as_index=False)['Tot_spend'].sum()
wholesale.groupby(['Channel'],as_index=False)['Tot_spend'].sum()

##1.2 There are 6 different varieties of items are considered. 
##Do all varieties show similar behaviour across Region and Channel?  
##Provide justification for your answer

df5=wholesale.groupby(['Channel'],as_index=False)['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen'].sum()
##sns.barplot(x=df5['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen'],y=df5[''])
df5.plot.bar(stacked=True)
##plt.legend(title='Abhinav')
plt.show()
df5.describe()


##1.3 On the basis of a descriptive measure of variability, which item shows the most inconsistent behaviour? Which items show the least inconsistent behaviour?

wholesale.var()
fresh=wholesale['Fresh']
fresh_arr=np.array(fresh.values.tolist())
variation(fresh_arr,axis=None)

milk=wholesale['Milk']
milk_arr=np.array(milk.values.tolist())
variation(milk_arr,axis=None)

grocery=wholesale['Grocery']
grocery_arr=np.array(grocery.values.tolist())
variation(grocery_arr,axis=None)

frozen=wholesale['Frozen']
frozen_arr=np.array(frozen.values.tolist())
variation(frozen_arr,axis=None)


Detergent=wholesale['Detergents_Paper']
Detergent_arr=np.array(Detergent.values.tolist())
variation(Detergent_arr,axis=None)


Delicatessen=wholesale['Delicatessen']
Delicatessen_arr=np.array(Delicatessen.values.tolist())
variation(Delicatessen_arr,axis=None)




##Fresh seems to be higly varying              1.600e+08 
##Delicatessen -seems to be least varying       7.953e+06

##1.4 Are there any outliers in the data?

sns.boxplot(y=wholesale['Fresh'])
sns.boxplot(y=wholesale['Milk'])
sns.boxplot(y=wholesale['Grocery'])
sns.boxplot(y=wholesale['Frozen'])
sns.boxplot(y=wholesale['Detergents_Paper'])
sns.boxplot(y=wholesale['Delicatessen'])

df=wholesale[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']]
ax=df.boxplot(figsize=(30,20))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
plt.tight_layout()
plt.show()

##Outliers are present in purchases of all 6 products 






##1.5 On the basis of your analysis, 
##what are your recommendations for the business? 
##How can your analysis help the business to solve its problem? 
##Answer from the business perspective

 
####################################

##Problem 2 CMSU data 

cmsu=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\Survey-1.csv")

df6=cmsu.copy()

##2.1.1. Gender and Major
##2.1.2. Gender and Grad Intention
##2.1.3. Gender and Employment
##2.1.4. Gender and Computer
a=pd.crosstab(df6['Gender'],df6['Major'],margins=True)
b=pd.crosstab(df6['Gender'],df6['Grad Intention'],margins=True)
c=pd.crosstab(df6['Gender'],df6['Employment'],margins=True)
e=pd.crosstab(df6['Gender'],df6['Computer'],margins=True)


##2.2.1. What is the probability that a randomly selected CMSU student will be male?

##p(Male)=No. Of males /Total no. of people
p_male=29/62

##2.2.2. What is the probability that a randomly selected CMSU student will be female?
##p(FeMale)=No. Of Females /Total no. of people

p_female=33/62

##2.3.1. Find the conditional probability of different majors among the male students in CMSU.

##Given that chosen student is a male ,we have to calc probs of diff majors,sample space becomes smaller ie 29 instead of 62
##P(Accounting | Male)
p_acc_given_male=4/29
p_cis_givenmale=1/29
p_eco_givenmale=4/29
p_intbus_givenmale=2/29
p_mgmnt_givenmale=6/29
p_oth_givenmale=4/29
p_ret_givenmale=5/29


##Given that chosen student is a female ,we have to calc probs of diff majors,sample space becomes smaller ie 33 instead of 62
##P(Accounting | FeMale)

p_acc_given_female=3/33
p_cis_given_female=3/33


##2.4.1. Find the probability That a randomly chosen student is a male and intends to graduate.

##p(male inter grad=Yes)

p_male_and_grd=17/62

##2.4.2 Find the probability that a randomly selected student is a female and does NOT have a laptop. 

##Prob(female INT Laptop=No)=P(female) * P(Laptop=No Given chosen person is female)
##=33/62 *4/33=4/62
p_female_and_laptopNo=4/62

##2.5.1. Find the probability that a randomly chosen student is either a male or has full-time employment?

##P(M or Emp=Full)=P(M)+P(Emp=Full)+P(M and Emp=Full)

p_male_orEmpFull=29/62+10/62-7/62

##2.5.2. Find the conditional probability that given a female student is randomly chosen, she is majoring in international business or management.

##As its given that a female is chosen ,the sample space is now 33 instead of 62
##P(IB Or Mgmnt)=P(IB)+P(Mgmnt)+P(IB AND Mgmnt)
p_IB_Or_Mgmnt=4/33+4/33+0

##2.6.  Construct a contingency table of Gender and Intent to Graduate at 2 levels (Yes/No). The Undecided students are not considered now and the table is a 2x2 table. Do you think the graduate intention and being female are independent events?

b.columns

##Drop the Undecided column from the dataframe 

b.drop(labels='Undecided',axis=1,inplace=True)

##H0-gender has nothing to do with Grad Intent 
##Ha -Gender has correlation with grad intent 
##Testing with Chi 2 at 5% significance level

chi2, pval, dof, exp_freq = chi2_contingency(b, correction = False)

pval 
##pval >0.05 Null hypo is accepted Gender has no relation with Graduation Intent


##2.7. Note that there are four numerical (continuous) variables in the data set, GPA, Salary, Spending, and Text Messages.
##2.6.1. If a student is chosen randomly, what is the probability that his/her GPA is less than 3?



f=cmsu.loc[(cmsu['GPA']<3)]

count_GPA_lessthan3=f['ID'].count()


p_gpa_lessthn3=count_GPA_lessthan3/62
p_gpa_lessthn3

##2.6.2. Find the conditional probability that a randomly selected male earns 50 or more. Find the conditional probability that a randomly selected female earns 50 or more.

only_male=cmsu.loc[(cmsu['Gender']=='Male')]

e=only_male.loc[(only_male['Salary']>=50)]

f=e['ID'].count()

p_male_50ormore=f/29


only_female=cmsu.loc[(cmsu['Gender']=='Female')]

h=only_female.loc[(only_female['Salary']>=50)]

i=e['ID'].count()

p_female_50ormore=i/33


##2.8. Note that there are four numerical (continuous) variables in the data set, GPA, Salary, Spending, and Text Messages. For each of them comment whether they follow a normal distribution. Write a note summarizing your conclusions.

plt.scatter(x=cmsu['ID'],y=cmsu['GPA'],c="black",marker="*")
plt.scatter(x=cmsu['ID'],y=cmsu['Salary'],c="black",marker="*")
plt.scatter(x=cmsu['ID'],y=cmsu['Spending'],c="black",marker="*")
plt.scatter(x=cmsu['ID'],y=cmsu['Text Messages'],c="black",marker="*")

##Based on simple observation none are following normal distribution


##
shingles=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\A+&+B+shingles.csv")

##3.1 Do you think there is evidence that means moisture contents in both types of shingles are within the permissible limits? State your conclusions clearly showing all steps.

A_sample=shingles['A']

B_sample=shingles['B']

##Step 1: Define null and alternative hypotheses 
##HA-MMC <0.35 /100sqft 
##H0 -MMC>=0.35/100 sqft 

##Step 2: Decide the significance level
##Alpha =0.05 

##Step 3 ::test statistic  --1 sample t test repeated over both the sample 

##Step 4: Calculate the p - value and test statistic
t_statistic, p_value = ttest_1samp(A_sample, 0.35)
print('One sample t test A Sample \nt statistic: {0} p value: {1} '.format(t_statistic, p_value))

t_statistic, p_value = ttest_1samp(B_sample, 0.35,nan_policy='omit')
print('One sample t test B Sample \nt statistic: {0} p value: {1} '.format(t_statistic, p_value))


##Step 5 Decide to reject or accept null hypothesis
##pval =0.14 >0.05 hence we fail to reject null hyp on A sample
##pval=0.00418 <0.05 hence null hyp is rejected in the 2nd case 

##3.2 Do you think that the population mean for shingles A and B are equal? Form the hypothesis and conduct the test of the hypothesis. What assumption do you need to check before the test for equality of means is performed?

##Step 1: Define null and alternative hypotheses
##H0:Sample A mean EQUAL to Sample B mean 
##HA:Sample A Mean NOT EQUAL to Sample B Mean 

##Step 2: Decide the significance level
##Here we select $\alpha$ = 0.05 and the population standard deviation is not known.

##Step 3: Identify the test statistic
##* We have two samples and we do not know the population standard deviation.
##* Sample sizes for both samples are  same.
##* The sample is not a large sample, n < 30. So you use the t distribution and the $t_{STAT}$ test statistic for two sample unpaired test.


##Step 4: Calculate the p - value and test statistic

t_statistic, p_value  = ttest_ind(A_sample,B_sample,nan_policy='omit')
print('tstat',t_statistic)    
print('P Value',p_value)  

##pval > 0.05 hence Null hypothesis is not rejected (accepted )ie sample A mean equal to sample B mean 

ci=pd.read_excel("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\Compound_Interest_Self.xlsx")

years=ci['Year'].tolist() 
amounts=ci['Amount'].tolist()
plt.plot(years,amounts,color='g',linestyle='solid',linewidth=3)
plt.xlabel("Years")
plt.ylabel("Compounded Amount")
plt.grid(True)
plt.show()


##qus 5 Which item contributes maximum to the Sodium intake?

b=df.groupby(['Item'],as_index=False)[['Sodium','Sodium (% Daily Value)']].sum().sort_values(by='Sodium',ascending=False).head(1)

ax4=sns.barplot(y=b['Item'],x=b['Sodium'])
ax4.set_xticklabels(
    ax4.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

