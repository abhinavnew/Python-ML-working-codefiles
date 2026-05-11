# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:27:24 2025

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


workflow_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Tenant_Details_OutputFile_workflow.csv')

workflow=workflow_orig.copy()


# Add the PRD/NON-PRD column based on TenantId
workflow['PRD/NON-PRD'] = workflow['TenantId'].apply(
    lambda x: 'PROD' if (isinstance(x, str) and '_prd_' in x) else 'NON-PROD'
)

# Display some information about the data
print(f"Total rows: {len(workflow)}")
prod_count = (workflow['PRD/NON-PRD'] == 'PROD').sum()
non_prod_count = (workflow['PRD/NON-PRD'] == 'NON-PROD').sum()
print(f"PROD count: {prod_count}")
print(f"NON-PROD count: {non_prod_count}")

# Display a sample of the modified data
print("\nSample of modified data (first 5 rows):")
print(workflow.head())

# Save to new CSV file
output_path = r'C:\Abhinav B\Product management\Extensions Adoption Data\Tenant_Details_with_PRD_Status.csv'
workflow.to_csv(output_path, index=False)
print(f"\nFile saved successfully as: {output_path}")



ipack_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Tenant_Details_OutputFile iPacks.csv')
ipack=ipack_orig.copy()


# Add the PRD/NON-PRD column based on TenantId
ipack['PRD/NON-PRD'] = ipack['TenantId'].apply(
    lambda x: 'PROD' if (isinstance(x, str) and '_prd_' in x) else 'NON-PROD'
)

# Display some information about the data
print(f"Total rows: {len(ipack)}")
prod_count = (ipack['PRD/NON-PRD'] == 'PROD').sum()
non_prod_count = (ipack['PRD/NON-PRD'] == 'NON-PROD').sum()
print(f"PROD count: {prod_count}")
print(f"NON-PROD count: {non_prod_count}")

# Display a sample of the modified data
print("\nSample of modified data (first 5 rows):")
print(ipack.head())

# Save to new CSV file
output_path = r'C:\Abhinav B\Product management\Extensions Adoption Data\Tenant_Details_iPacks_with_PRD_Status.csv'
ipack.to_csv(output_path, index=False)
print(f"\nFile saved successfully as: {output_path}")





merge_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Usage_Merged.csv')
merge=merge_orig.copy()




import re

# Function to extract the core feature name
def extract_feature_name(name):
    if not isinstance(name, str):
        return name
        
    # Split by underscore and take the first part
    name_without_underscore = name.split('_')[0]
    
    # Remove numbers and special characters
    # Keep only letters, spaces and some special characters like "Timecard"
    feature_name = re.sub(r'[0-9]', '', name_without_underscore)
    
    # Further clean up: remove any remaining special characters like colons
    feature_name = re.sub(r'[:;].*$', '', feature_name)
    
    # Handle specific cases like "TimecardApproval"
    if "TimecardApproval" in feature_name:
        return "TimecardApproval"
    elif "Timecard" in feature_name:
        return "Timecard"
    elif "GTOR" in feature_name:
        return "GTOR"
    
    return feature_name

# Apply the function to create the new Feature column
merge['Feature'] = merge['Name'].apply(extract_feature_name)

# Display information about the data
print(f"Total rows: {len(merge)}")

# Show unique features extracted and their counts
feature_counts = merge['Feature'].value_counts()
print("\nFeature counts:")
print(feature_counts)

# Display a sample of the modified data
print("\nSample of modified data (first 5 rows):")
print(merge.head())

# Save to new CSV file
output_path = r'C:\Abhinav B\Product management\Extensions Adoption Data\Usage_Merged_with_Feature.csv'
merge.to_csv(output_path, index=False)
print(f"\nFile saved successfully as: {output_path}")





feature_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Usage_Merged_with_Feature.csv')
feature=feature_orig.copy()


# Display info about the original data
print("Original dataset shape:", feature.shape)
print("Sample before updates:")
print(feature[['Name', 'Feature']].head(10))

# First, let's create a mask for each condition to make the code more readable
gtor_mask = feature['Name'].str.contains('GTOR', case=False, na=False)
gtor_doc_mask = feature['Name'].str.contains('GTOR.*Document|GTOR.*Attachment', case=False, na=False, regex=True)

# Update Feature column for rows where Name contains GTOR Document or Attachment
feature.loc[gtor_doc_mask, 'Feature'] = 'GTORDocumentAttachment'

# Update Feature column for rows where Name contains GTOR (but not already updated)
# This needs to come after the more specific rule
feature.loc[gtor_mask & ~gtor_doc_mask, 'Feature'] = 'GTOR_Validations'

# Display counts of each Feature value after updates
print("\nFeature value counts after updates:")
print(feature['Feature'].value_counts().head(10))

# Display sample after updates to verify the changes
print("\nSample after updates:")
print(feature[['Name', 'Feature']].head(10))

# Save to new CSV file
output_path = r'C:\Abhinav B\Product management\Extensions Adoption Data\Usage_Merged_with_Updated_Feature_new.csv'
feature.to_csv(output_path, index=False)
print(f"\nFile saved successfully as: {output_path}")


















