# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 22:59:51 2020

@author: Abhinav.Bajpai
"""

""
import numpy as np 
import pandas as pd
import matplotlib as mp
import seaborn as sns
import os
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from   scipy.stats import ttest_1samp, ttest_ind

sns.set(color_codes=True)



startup=pd.read_csv(r"C:\Users\Abhinav.Bajpai\Downloads\startup_funding.csv")

startup["Amount in USD"].max()

df1=startup.copy()

##Startup details with the highest funding in USD in any year 
df1.loc[(df1["Amount in USD"]==df1["Amount in USD"].max())]


sns.barplot(df1['month_funding'],df1['Amount in USD']);

sns.barplot(df1['Amount in USD'],df1['Investors Name']);

sns.boxplot(df1['year_funding'],df1['Amount in USD']);

sns.countplot(df1['Industry Vertical']);

##Year Wise highest funded startup details 
df2=df1.groupby(['year_funding'],sort=False)['Amount in USD'].max()
print(df2)

##Queries 

##sns.countplot(df1['Industry Vertical']); qus ::how to correct the text labels on x axis ?
##sns.boxplot(df1['year_funding'],df1['Amount in USD']);-->QUS :: Y axis having values in e^n notation how to correct that ?


##Find Year with funding of max no of startups count
df4=df1.groupby('year_funding')['Startup Name'].transform('nunique')

plt.figure(figsize = (15,6));

sns.countplot(df1['InvestmentnType']);


###Batting analysis 

batsman=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week3\\Batsman.csv")


pakistan=batsman.loc[(batsman['Opposition']=='Pakistan')]


Q1=pakistan["RunsScored"].quantile(.25)


Q3=pakistan["RunsScored"].quantile(.75)

IQR=Q3-Q1

max=Q3+(1.5 * IQR)

print(max)


pd.crosstab(df[],df[],margins=True)
ct.drop('All',axis=1,inplace=True)


## calc nCr effectively 
ncr=math.factorial(50)/(math.factorial(25)*math.factorial(25))
p1=0.5**25
fin=ncr*p1*p1
print(fin)

##Area of normal dist curve based on z value
##Stats.norm.cdf(z -value) or stats.norm.cdf(val asked for ,mean,sdx)

stats.norm.cdf(-0.6)
stats.norm.cdf(-3.016)

z_60=(60-65.16)/10
z_50=(50-65.16)/10
z_40=(40-65.16)/10
1-stats.norm.cdf(z_60)
stats.norm.cdf(z_50)-stats.norm.cdf(z_40)



z_val1=(248.5-250)/(4.8/6)

z_val1


stats.norm.cdf(z_val1)

##qus 5 Which item contributes maximum to the Sodium intake?

b=df.groupby(['Item'],as_index=False)[['Sodium','Sodium (% Daily Value)']].sum().sort_values(by='Sodium',ascending=False).head(1)

ax4=sns.barplot(y=b['Item'],x=b['Sodium'])
ax4.set_xticklabels(
    ax4.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')




##Poisson dist

rate=6
n=np.arange(0,200)

poi=stats.poisson.pmf(n,rate)

n

poi

poi[4]

plt.plot(n,poi,'o-')


##Binomial dist
##k->value of the random variable for whihc we are calc the probability ;n->trials;p->probab of success

n_trials=100

p_succ=0.1

k=np.arange(0,101)

bin_dist=stats.binom.pmf(k,n_trials,p_succ)
plt.title('Binomial chart')
plt.xlabel('No of defective laptops')
plt.ylabel('Probabilties')
plt.plot(k,bin_dist,'o-')

stats.binom.pmf(0,7,0.75)+stats.binom.pmf(1,7,0.75)+stats.binom.pmf(2,7,0.75)+stats.binom.pmf(3,7,0.75)

stats.binom.cdf(3,7,0.75)

##hypothesis testing 

#Storing the wait time sample data into object mwt.
mwt=[4.21, 5.55, 3.02, 5.13, 4.77, 2.34, 3.54, 3.20, 4.50, 6.10, 0.38, 5.12, 6.46, 6.19, 3.79]

from scipy.stats import ttest_1samp
t_statistic, p_value = ttest_1samp(mwt,5)
t_statistic, p_value


from scipy.stats import shapiro
stat, p = shapiro(mwt)
stat, p # Since p (0.5085)> alpha (0.05), the sample looks Gaussian (Normal Distribution)


##2 sample independent t test 

mydata = pd.read_csv('Luggage.csv')
mydata.head()

t_statistic, p_value  = ttest_ind(mydata['WingA'],mydata['WingB'])
print('tstat',t_statistic)    
print('P Value',p_value)    

# p_value < 0.05 => alternative hypothesis:
# they don't have the same mean at the 5% significance level
print ("two-sample t-test p-value=", p_value)

alpha_level = 0.05

if p_value < alpha_level:
    print('We have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
    print('We conclude that the mean time to deliver luggages in of both the wings of the hotel are not same.')
else:
    print('We do not have enough evidence to reject the null hypothesis in favour of alternative hypothesis')
    print('We conclude that mean time to deliver luggages in of both the wings of the hotel are same.')

##Trough quiz questions


##Step 1 :Frame hypothesis 
##Step 2 : Decide the level of significance
##Step 3 : Decide on which test to perform to test hypothesis Z test ,t test , 2 sample t test,paired t test
##Step 4 : Find P value 
##Step 5 :Compare p-value and Alpha ,if p-value <Alpha then reject the NULL Ho hypothesis

trough=pd.read_excel("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week6 -hypothesis\\Trough.xlsx")


print("The sampe size of the problem is ",len(trough))


t_stat,p_val=ttest_1samp(trough, 8.46)

cars_quiz=pd.read_excel("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week6 -hypothesis\\Quiz_Dataset.xlsx")


t_stat,p_val=ttest_ind(cars_quiz['US_Cars'],cars_quiz['Japanese_Cars'], nan_policy='omit')

##Chi sq test for categorical values

from   scipy.stats        import    chi2_contingency


df = pd.DataFrame({'Highschool': [60, 40], 'Bach': [54, 44],'Mast':[46,53],'phd':[41,57]},index = ['Female', 'Male'])

chi2, pval, dof, exp_freq = chi2_contingency(df, correction = False)
pval
chi2

((-10*5)/40)

stats.norm.cdf()