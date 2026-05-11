# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:23:54 2020

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


mcd=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\Mcdonald .csv")

df=mcd.copy()

##Plot graphically which food categories have the highest and lowest varieties

a=df.groupby(['Category'],as_index=False).size().reset_index(name="Count of All Obeservations/Freq")

plt.figure(figsize=(50,60))
plt.hist(mcd['Category'])
##plt.xticks(rotation=45)

sns.countplot(mcd['Category']
              
  ## Which all variables have an outlier?           
ax=df.boxplot(figsize=(40,10))
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right', fontsize=12)
plt.tight_layout()
plt.show()

##Plot shows all values have outliers except Saturated fat ,Saturated Fat (% Daily Value),Dietary fibre


##Qus 3 Which variables have the highest correlation? Plot them and find out the value?

corrmat=df.corr()
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

corrmat.to_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\corrmat.csv")
##Answer calories/calories from FAT/Total Fat /Total Fat (% Daily value)/Saturated Fat/Saturated Fat (%Daily Value)/Sodium /Sodium (%Daily Value)/Protein/Iron 
##Saturated Fat/Saturated Fat (% Daily value) 
##Cholestrol /Cholestrol (% Daily Value)
##Sodium /Sodium (%Daily Value)
##Carbohydrates/Carbohydrates(%daily Value)
##Dietary fibre/Dietary fibre(%Daily value)

##qus 4 :Which category contributes to the maximum % of Cholesterol in a diet (% daily value)?

df.groupby(['Category'],as_index=False)['Cholesterol (% Daily Value)'].sum().sort_values(by='Cholesterol (% Daily Value)',ascending=False)

a=df.groupby(['Category'],as_index=False)['Cholesterol (% Daily Value)'].sum()

ax3=sns.barplot(x=a['Category'],y=a['Cholesterol (% Daily Value)'])
ax3.set_xticklabels(
    ax3.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')


##qus 5 Which item contributes maximum to the Sodium intake?

b=df.groupby(['Item'],as_index=False)[['Sodium','Sodium (% Daily Value)']].sum().sort_values(by='Sodium',ascending=False).head(1)

ax4=sns.barplot(y=b['Item'],x=b['Sodium'])
ax4.set_xticklabels(
    ax4.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

##qus 6 Which 4 food items contain the most amount of Saturated Fat?

c=df.groupby(['Item'],as_index=False)['Saturated Fat'].sum().sort_values(by='Saturated Fat',ascending=False).head(4)
