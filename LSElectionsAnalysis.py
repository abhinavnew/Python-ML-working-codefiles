# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:17:11 2020

@author: Abhinav.Bajpai
"""

import numpy as np 
import pandas as pd
import matplotlib as mp
import seaborn as sns
import os
import matplotlib.pyplot as plt
sns.set(color_codes=True)
%matplotlib inline 


# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)


pd.set_option("display.max_colwidth",10)

loksabha=pd.read_csv("C:\\Users\\Abhinav.Bajpai\\Downloads\\Lok+Sabha_2019.csv")

print(loksabha.shape)
print(loksabha.columns)
print(loksabha.dtypes)
print(loksabha.head(5))

df=loksabha.copy()
print(df.shape)



df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df.shape)
print(df.columns)

##Theme 1 
##Candidate namewise education qual count 
df.groupby('NAME')['EDUCATION'].nunique()

##Overall   unique Educational qualf
df['EDUCATION'].nunique()


##Theme 2

##Count of winners -top5 parties 
a=df.groupby(['PARTY'],as_index=False).agg({'WINNER':'sum'}).sort_values(by='WINNER',ascending=False).head(5)
print(a)

##Find the Male: Female ratio in the top 5 parties which have the maximum number of winners.
##Using "a" in the above query 
b=a['PARTY'].tolist()
dftop5=df.loc[df['PARTY'].isin(b)]
dftop5.groupby(['PARTY','GENDER'],as_index=False).size().reset_index(name="Count Of M/F")


##count of candidates from different categories for the top 5 parties which have the maximum number of winners
dftop5.groupby(['PARTY','CATEGORY'],as_index=False).size().reset_index(name="Count Of Diff Cat.Candidates")


##average age of the candidates for the top 5 parties
dftop5.groupby(['PARTY'],as_index=False).agg({'AGE':'mean'})



##Theme 3
##top 5 states with the maximum number of constituencies
c=df.groupby(['STATE'],as_index=False)['CONSTITUENCY'].count().sort_values(by='CONSTITUENCY',ascending=False).head(5)
print(c)

##find the average number of criminal cases per candidate.
d=c['STATE'].tolist()
top5state=df.loc[df['STATE'].isin(d)]
top5state.groupby(['STATE'],as_index=False)['CRIMINAL CASES'].mean()


##find the average value of the assets per candidate.
top5state.groupby(['STATE'],as_index=False)['ASSETS'].mean()


##Theme 4
##overall average value of assets of the winners and the losers of the election
e=df.groupby(['WINNER'],as_index=False)['ASSETS'].mean()
##ratio of the overall average value of assets of the Winners and Losers of the election
winner_avg_assets=e.loc[(e['WINNER']==1),'ASSETS'].tolist()
loser_avg_assets=e.loc[(e['WINNER']==0),'ASSETS'].tolist()
print(winner_avg_assets[0]/loser_avg_assets[0])


##overall average number of criminal cases for the winners and the losers of the election
f=df.groupby(['WINNER'],as_index=False)['CRIMINAL CASES'].mean()
##ratio of the overall average number of criminal cases of the Winners and Losers of the election
winner_avg_cases=f.loc[(f['WINNER']==1),'CRIMINAL CASES'].tolist()
loser_avg_cases=f.loc[(f['WINNER']==0),'CRIMINAL CASES'].tolist()
print(winner_avg_cases[0]/loser_avg_cases[0])

##overall gender distribution (number of Males and Females) for both the winners and the losers of the election
df.groupby(['WINNER','GENDER'],as_index=False).size().reset_index(name="Count Of M/F Candidates")

##What is the overall average age for the winners and the losers of the election
g=df.groupby(['WINNER'],as_index=False)['AGE'].mean()
print(g)

##find the ratio of the overall average age of the Winners and the Losers in the election.
winner_avg_age=g.loc[(g['WINNER']==1),'AGE'].tolist()
loser_avg_age=g.loc[(g['WINNER']==0),'AGE'].tolist()
print(winner_avg_age[0]/loser_avg_age[0])


##Theme5
##What was the percentage of votes received by Mr Narendra Modi in the Varanasi constituency

varanasi=df.loc[(df['CONSTITUENCY']=='VARANASI') & (df['WINNER']==1)]
print (varanasi)
varanasi['PctOfVotes']=varanasi['TOTAL VOTES']/varanasi['OVER TOTAL VOTES POLLED IN CONSTITUENCY']
print (varanasi)


##candidate received the highest percentage of votes

df['PctVotes']=(df['TOTAL VOTES']/df['OVER TOTAL ELECTORS IN CONSTITUENCY'])*100
h=df.sort_values(by='PctVotes',ascending=False).head(1)
h['NAME']

##Name the 2 candidates who fought elections in Gandhinagar Constituency

gandhinagar=df.loc[(df['CONSTITUENCY']=='GANDHINAGAR')]
gandhinagar['NAME']

##Density distribution plot 
sns.distplot(df['CRIMINAL CASES']);


##Scatter plot

sns.jointplot(df['STATE'], df['EDUCATION']);

sns.jointplot(df['NAME'], df['TOTAL VOTES']);

sns.boxplot(df['STATE'],df['TOTAL VOTES']);






