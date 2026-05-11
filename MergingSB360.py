# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 20:32:44 2025

@author: Abhinav.Bajpai
"""


import sys
print(sys.executable)

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
print('The seaborn version is {}.'.format(sns.__version__))
import os
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import zscore
import datetime 
from scipy.stats import chi2_contingency ## for chi sq test 
from scipy.stats import   ttest_1samp,ttest_ind ##For t test
from scipy.stats import variation  
from statsmodels.formula.api import ols      # For n-way ANOVA
from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA
from sklearn.preprocessing import StandardScaler ##For zscore scaling 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster ##Hierachical clustering
from sklearn.metrics import silhouette_samples, silhouette_score  ##kmeans sil score
from sklearn.tree import DecisionTreeClassifier   ##DT for CART
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve
 ##for clustering
import time
from datetime import timedelta
from sklearn.cluster import KMeans  ## for k means
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.metrics import make_scorer
sns.set(color_codes=True) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
import sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import re
import xgboost as xgb
from xgboost import plot_importance
import pickle
import glob
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))




##
# Argentina
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Argentina.xlsx')
argentina_df = pd.read_excel(xls, sheet_name='Export')
argentina_df['Country'] = "Argentina"
print(argentina_df.head(5))
print(len(argentina_df))

# Australia
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Australia.xlsx')
australia_df = pd.read_excel(xls, sheet_name='Export')
australia_df['Country'] = "Australia"
print(australia_df.head(5))
print(len(australia_df))

# Belgium
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Belgium.xlsx')
belgium_df = pd.read_excel(xls, sheet_name='Export')
belgium_df['Country'] = "Belgium"
print(belgium_df.head(5))
print(len(belgium_df))

# Brazil
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Brazil.xlsx')
brazil_df = pd.read_excel(xls, sheet_name='Export')
brazil_df['Country'] = "Brazil"
print(brazil_df.head(5))
print(len(brazil_df))

# Canada
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Canada.xlsx')
canada_df = pd.read_excel(xls, sheet_name='Export')
canada_df['Country'] = "Canada"
print(canada_df.head(5))
print(len(canada_df))

# Chile
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Chile.xlsx')
chile_df = pd.read_excel(xls, sheet_name='Export')
chile_df['Country'] = "Chile"
print(chile_df.head(5))
print(len(chile_df))

# Colombia 
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Coloumbia.xlsx')
colombia_df = pd.read_excel(xls, sheet_name='Export')
colombia_df['Country'] = "Colombia"
print(colombia_df.head(5))
print(len(colombia_df))

# Costa Rica
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Costa Rica.xlsx')
costa_rica_df = pd.read_excel(xls, sheet_name='Export')
costa_rica_df['Country'] = "Costa Rica"
print(costa_rica_df.head(5))
print(len(costa_rica_df))

# Denmark
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Denmark.xlsx')
denmark_df = pd.read_excel(xls, sheet_name='Export')
denmark_df['Country'] = "Denmark"
print(denmark_df.head(5))
print(len(denmark_df))

# France
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-France.xlsx')
france_df = pd.read_excel(xls, sheet_name='Export')
france_df['Country'] = "France"
print(france_df.head(5))
print(len(france_df))

# Germany
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Germany.xlsx')
germany_df = pd.read_excel(xls, sheet_name='Export')
germany_df['Country'] = "Germany"
print(germany_df.head(5))
print(len(germany_df))

# Ireland
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Ireland.xlsx')
ireland_df = pd.read_excel(xls, sheet_name='Export')
ireland_df['Country'] = "Ireland"
print(ireland_df.head(5))
print(len(ireland_df))

# Japan
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Japan.xlsx')
japan_df = pd.read_excel(xls, sheet_name='Export')
japan_df['Country'] = "Japan"
print(japan_df.head(5))
print(len(japan_df))

# Luxembourg 
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Luxomberg.xlsx')
luxembourg_df = pd.read_excel(xls, sheet_name='Export')
luxembourg_df['Country'] = "Luxembourg"
print(luxembourg_df.head(5))
print(len(luxembourg_df))

# Mexico
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Mexico.xlsx')
mexico_df = pd.read_excel(xls, sheet_name='Export')
mexico_df['Country'] = "Mexico"
print(mexico_df.head(5))
print(len(mexico_df))

# Netherlands
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Netherlands.xlsx')
netherlands_df = pd.read_excel(xls, sheet_name='Export')
netherlands_df['Country'] = "Netherlands"
print(netherlands_df.head(5))
print(len(netherlands_df))

# New Zealand (fixed spacing)
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Newzealand.xlsx')
new_zealand_df = pd.read_excel(xls, sheet_name='Export')
new_zealand_df['Country'] = "New Zealand"
print(new_zealand_df.head(5))
print(len(new_zealand_df))

# Panama
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Panama.xlsx')
panama_df = pd.read_excel(xls, sheet_name='Export')
panama_df['Country'] = "Panama"
print(panama_df.head(5))
print(len(panama_df))

# Poland
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Poland.xlsx')
poland_df = pd.read_excel(xls, sheet_name='Export')
poland_df['Country'] = "Poland"
print(poland_df.head(5))
print(len(poland_df))

# Saudi Arabia
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Saudi Arabia.xlsx')
saudi_arabia_df = pd.read_excel(xls, sheet_name='Export')
saudi_arabia_df['Country'] = "Saudi Arabia"
print(saudi_arabia_df.head(5))
print(len(saudi_arabia_df))

# Singapore
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Singapore.xlsx')
singapore_df = pd.read_excel(xls, sheet_name='Export')
singapore_df['Country'] = "Singapore"
print(singapore_df.head(5))
print(len(singapore_df))

# Spain
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Spain.xlsx')
spain_df = pd.read_excel(xls, sheet_name='Export')
spain_df['Country'] = "Spain"
print(spain_df.head(5))
print(len(spain_df))

# Switzerland
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-Switzerland.xlsx')
switzerland_df = pd.read_excel(xls, sheet_name='Export')
switzerland_df['Country'] = "Switzerland"
print(switzerland_df.head(5))
print(len(switzerland_df))

# UAE
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-UAE.xlsx')
uae_df = pd.read_excel(xls, sheet_name='Export')
uae_df['Country'] = "UAE"
print(uae_df.head(5))
print(len(uae_df))

# UK
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-UK.xlsx')
uk_df = pd.read_excel(xls, sheet_name='Export')
uk_df['Country'] = "UK"
print(uk_df.head(5))
print(len(uk_df))

# USA
xls = pd.ExcelFile(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Contract Details-USA.xlsx')
usa_df = pd.read_excel(xls, sheet_name='Export')
usa_df['Country'] = "USA"
print(usa_df.head(5))
print(len(usa_df))


# Concatenate all dataframes into one
all_countries = [
    argentina_df, australia_df, belgium_df, brazil_df, canada_df, 
    chile_df, colombia_df, costa_rica_df, denmark_df, france_df, 
    germany_df, ireland_df, japan_df, luxembourg_df, mexico_df, 
    netherlands_df, new_zealand_df, panama_df, poland_df, saudi_arabia_df, 
    singapore_df, spain_df, switzerland_df, uae_df, uk_df, usa_df
]

combined_df = pd.concat(all_countries, ignore_index=True)
print(f"Combined dataframe has {len(combined_df)} rows")
print(combined_df['Country'].value_counts())

# Save combined dataframe
combined_df.to_csv(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Combined_Country_Data.csv', index=False)
combined_df.to_excel(r'C:\Abhinav B\Product management\SB 360 data\dashboard\Combined_Country_Data.xlsx', index=False)














