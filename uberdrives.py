# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:35:18 2020

@author: Abhinav.Bajpai
"""

import numpy as np 
import pandas as pd
import matplotlib as mp
import seaborn as sns
import os
import matplotlib.pyplot as plt
sns.set(color_codes=True)
##%matplotlib inline 


# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)

uber=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\Week3\\uberdrive-1.csv")

##Show the last 10 records of the dataset. (2 point)
uber.tail(10)
##Show the first 10 records of the dataset. (2 points)
uber.head(10)
##Q3. Show the dimension(number of rows and columns) of the dataset. (2 p
uber.shape
## Show the size (Total number of elements) of the dataset. (2 points)
uber.size
##Q5. Display the information about all the variables of the data set. 
uber.info()
##6. Check for missing values. (2 points) - Note: Output should be boolean only
uber.isna()
##How many missing values are present? (2 points)
uber.isna().sum()
##Get the summary of the original data. (2 points)
uber.describe()
##Drop the missing values and store data in a new dataframe (name it"df") (2-points)

df=uber.dropna(axis=0,how="any")

##. Check the information of the dataframe(df). (2 points)

df.info()

##Hint- You need to print the unique start locations place names in this and not the count.

df['START*'].unique()

##Q12. What is the total number of unique start locations? (2 points)--Original datafarme UBER
uber['START*'].nunique()

##Q13. What is the total number of unique stop locations. (2 points)

uber['STOP*'].nunique()

##Q14. Display all the Uber trips that has the starting point of San Francisco. (2 points)
uber.loc[(uber['START*']=='San Francisco')]
##Q15. What is the most popular starting point for the Uber drivers? (2 points)
uber['START*'].value_counts().head(1)
uber.groupby(['START*'],as_index=False).size().sort_values(ascending=False)

 ##What is the most popular dropping point for the Uber drivers? (2 points)
uber.groupby(['STOP*'],as_index=False).size().sort_values(ascending=False).head(1)

##Q17. List the most frequent route taken by Uber drivers. (3 points)

df.groupby(['START*','STOP*'],as_index=False).size().sort_values(ascending=False).head(1)

##Q18. Display all types of purposes for the trip in an array. (3 points

arr1=df['PURPOSE*']
print(arr1)

##Q19. Plot a bar graph of Purpose vs Miles(Distance). (3 points)
uber['PURPOSE*'].nunique()
df2=uber.groupby(['PURPOSE*'],as_index=False)['MILES*'].sum()
sns.barplot(y=df2['PURPOSE*'],x=df2['MILES*']);

##Q20. Display a dataframe of Purpose and the distance travelled for that particular Purpose. (3 points)
df2=uber.groupby(['PURPOSE*'],as_index=False)['MILES*'].sum()
print(df2)

##Plot number of trips vs Category of trips. (3 points)
sns.countplot(uber['CATEGORY*']);


##Q22.What is proportion of miles that are covered as Business trips 
##and what is the proportion of miles that are covered as Personal trips? (3 points)

tot_miles=uber['MILES*'].sum()
biz_miles=uber.loc[(uber['CATEGORY*']=='Business')]['MILES*'].sum()
personal_miles=uber.loc[(uber['CATEGORY*']=='Personal')]['MILES*'].sum()
Prop_biz_miles=biz_miles/tot_miles
Prop_per_miles=personal_miles/tot_miles
print(Prop_biz_miles)
print(Prop_per_miles)

##Defect counts

defects=pd.read_csv("C:\\Users\\Abhinav.Bajpai\\Downloads\\ENGJira 7.13.8 Production System 2020-12-09T03_33_09-0500.csv")

sns.barplot(defects['month_funding'],df1['Amount in USD']);

sns.barplot(df1['Amount in USD'],df1['Investors Name']);

sns.boxplot(df1['year_funding'],df1['Amount in USD']);

sns.countplot(y=defects['Reporter']);

a=pd.crosstab(defects['Resolution'],defects['Reporter'],margins=True)

b=defects.groupby(['Reporter'],as_index=False)['Issue id'].count()
b=defects.groupby(['Reporter'],as_index=False).agg({'Issue id','count'})
print(b)

sns.countplot(y=defects['Reporter'],hue=defects['Resolution']);

pd.crosstab(df[],df[],margins=True)
ct.drop('All',axis=1,inplace=True)



##Iris dataset 

iris=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week4 -Stats\\IRIS.csv")

iris.drop(iris.columns[iris.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)


iris.hist(figsize=(20,30));

sns.boxplot(x="sepal length",y="petal length",data=iris)


iris['sepal length'].mean()

iris['petal length'].std()

sns.pairplot(iris)

sns.boxplot(iris)

IQR_SW=iris['sepal width'].quantile(.75)-iris['sepal width'].quantile(.25)

iris['sepal width'].quantile(.75)+1.5*IQR_SW


IQR_SL=iris['sepal length'].quantile(.75)-iris['sepal length'].quantile(.25)

iris['sepal length'].quantile(.75)+1.5*IQR_SL

sp=(15000/212000)*100
tab=(1500/5020)*100
laptop=(4000/30000)*100

print(sp,tab,laptop)






