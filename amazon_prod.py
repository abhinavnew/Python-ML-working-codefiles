# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:21:04 2020

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
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)


pd.set_option("display.max_colwidth",10)


amazon=pd.read_csv("E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week3\\amazon_data.csv")

sns.pairplot(amazon[['mrp', 'sale price', 'discount percentage','number of reviews']],kind="kde");

sns.distplot(amazon['discount percentage'],hist=False)
plt.show()

plt.xticks(rotation=45)