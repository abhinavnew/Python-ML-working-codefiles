# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:46:54 2020

@author: Abhinav.Bajpai
"""

import numpy as np 
import pandas as pd
import matplotlib as mp
import seaborn as sns
import os
import datetime
import sklearn
import xgboost as xgb





gdp=pd.read_excel(r"C:\Users\Abhinav.Bajpai\Downloads\GDP_Dataset.xlsx")

print(gdp.shape)
print(gdp.columns)
print(gdp.dtypes)
print(gdp.head(5))

##Full row of country with highest GDP
gdp.loc[(gdp['GDP']==gdp['GDP'].max())]


##Status wise counts of total observations
gdp['Status'].value_counts()

##List countries with highest GDP but STATUS wise 
gdp.loc[gdp.groupby('Status')['GDP'].idxmax()]


demo_array=np.arange(0,10) 

demo_array
demo_array <3
demo_array[demo_array <6]
np.max(demo_array)

demo_matrix = np.array(([13,35,74,48], [23,37,37,38],[73,39,93,39]))
demo_matrix[:, (1,2)]


demo_array = np.arange(10,21)
subset_demo_array = demo_array[0:7]
subset_demo_array[:]= 101

subset_demo_array

a=gdp.head(5).copy()
newcolumn = ["A", "B", "C", "D", "E"]

a['Alphabets'] = newcolumn
print(a.columns)

gdp['Population'].sum()
gdp.loc[(gdp['Median_Age']==gdp['Median_Age'].max())]
gdp['Median_Age'].max()

gdp['PercentUByT']=(gdp['Urban_Population']/gdp['Population'])*100

gdp['PercentUByT']

gdp[['Country','Population_Density']].sort_values(by='Population_Density',ascending=False)

b=gdp['GDP'].loc[(gdp['Country']=="India")]

print(b)

gdp[['Country','GDP']].loc[(gdp['GDP']>2651)]


gdp[['Country','Population','GDP_per_Capita','GDP']].loc[(gdp['Population']>100000000) & (gdp['GDP']==250)]

