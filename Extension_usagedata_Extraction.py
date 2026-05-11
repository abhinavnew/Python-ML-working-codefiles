# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:09:37 2025

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
from difflib import get_close_matches
print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))




#############processing workflow file #####################


workflow_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Extenson adoption data - Oct 2025\Workflow_Oct_2025.csv')

workflow=workflow_orig.copy()
print(workflow.head())


print(workflow.head(5))
print(workflow.dtypes)
print(workflow.describe())
print(workflow.info())


# Remove hidden spaces or characters explicitly
workflow['Date'] = workflow['Date'].astype(str).str.strip()

# Convert to datetime explicitly specifying dayfirst=True
workflow['Date'] = pd.to_datetime(workflow['Date'], dayfirst=True, errors='coerce')

# Extract Month and Year correctly
workflow['Month'] = workflow['Date'].dt.month
workflow['Year'] = workflow['Date'].dt.year

# Display problematic rows if any
if workflow['Date'].isna().any():
    print("Problematic rows:")
    print(workflow[workflow['Date'].isna()])
else:
    print(workflow.head())

# Logic to create 'PRD/NON-PRD' column based on 'TenantId'
workflow['PRD/NON-PRD'] = workflow['TenantId'].apply(lambda x: 'NON-PROD' if 'nonprd' in x.lower() else ('PROD' if 'prd' in x.lower() else 'UNKNOWN'))

# Verify the DataFrame with the new columns
print(workflow.head())

# Add 'Type' column with constant value 'Workflow'
workflow['Type'] = 'Workflow'

# Verify the DataFrame with the new columns
print(workflow.head())


# List of all possible features
feature_list = [
    'HoursAndDollarsAllocation', 'ScheduleTemplateUpload', 'ProratedAccruals',
    'TransferReportToSFTP', 'PendingCorrectionsComputation', 'EnhancedStaffingDashboard',
    'ReligiousHolidayAdjustments', 'OpenShiftPayIncentive', 'PressGaneyNDNQIReport',
    'BrazilComplianceReports', 'FlexibleBreakAdjustment', 'MealPenalty',
    'SplitShiftPremiumandSpreadofHoursPay', 'PayPeriodMinimumWageTopUp',
    'DynamicPayCodeAllocationForActivities', 'SchedulePostAuditReport', 'MexicoXOvertime',
    'Mondayisation', 'ChileAuditReports', 'CWFMPeopleImport', 'CWFMImportExport', 
    'CWFMReports',
    'UltiProPeopleImport', 'JoursdeFractionnement', 'BetterOffOverallTest', 'MultiWeekWorkloadPattern',
    'LeaveLoading', 'MobilePunchTimeZoneAdjustment', 'GRIAWorkingDaysandWeekendsCompliance',
    'WorkingTimeDirectiveReport', 'SicknessPeriodManagement', 'AnonymizationTool',
    'AverageAbsenceCalculation', 'AbsenceBasedAccrualAdjustment', 'FranceVacationCalculation',
    'RightToRest', 
    'BrazilRoundingRules', 'FLSAiOvertimeExemption', 'GPTW', 'NYCOTCap',
    'OptimizedModulation', 'MonthlyVacationGrants', 'FinlandAccrualsCalculation',
    'PaidSickYearlyBonusPayout', 'EvidencePunchImport', 'EvidencePunchAttestation',
    'TimecardApproval', 'GTOR_Validations', 'GTORDocumentAttachment', 'TimecardValidation',
    'VolunteerToWork', 'AccrualPayoutRequest', 'ModifiedAccrualPayoutRequest', 'LeaveTimeRequest',
    'OpenShift', 'TotalizerExtensibilitySetup', 'DirectAccrualDonationv.',
    'DirectAccrualDonationCustomGroupv.', 'AccrualPool', 'ScheduleChangeAttestation',
    'PredictiveSchedulingAttestation', 'DirectAccrualDonationv', 'EmployeeManagedWorkLocations',
    'AutoShifts', 'DocumentAttachment', 'ShiftSwapWorkruleOverride', 'ShiftSwapFramework',
    'ShiftSwapWorkruleOverrideMainProcess', 'SchedulePostAcknowledgement',
    'MultipleReviewersSchedulePost', 'AccrualPool-Received', 'ReturnBasedVacationDeduction',
    'ChilePunchReceipt', 'AbsenceManagementv', 'DirectAccrualDonation', 'TimeOff',
     'GPTW_ProWFM', 'SalariedPresenceRecording', 'VolunteerToWork', 'AccrualPool'
]

# Matching logic function
def match_feature(name, feature_list):
    name_processed = name.replace(" ", "").lower()
    features_processed = [feature.lower() for feature in feature_list]
    matches = get_close_matches(name_processed, features_processed, n=1, cutoff=0.2)
    return feature_list[features_processed.index(matches[0])] if matches else 'Unknown'

# Add 'Feature' column based on fuzzy matching logic
workflow['Feature'] = workflow['Name'].apply(lambda x: match_feature(x, feature_list))

# Verify the DataFrame with the new columns
print(workflow.head())


# Check distinct values for column 'Feature'
print("Distinct values in 'Feature':", workflow['Feature'].unique())


# Add 'Data Extract date' column with constant value 'Oct-25'
workflow['Data Extract date'] = 'Oct-2025'

# Final verification of DataFrame --should be 9 columns
print(workflow.head())




#########################processing Boomi file ##############################################################################

boomi_orig=pd.read_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Extenson adoption data - Oct 2025\Boomi_Oct_2025.csv')

boomi=boomi_orig.copy()
print(boomi.head())


print(boomi.head(5))
print(boomi.dtypes)
print(boomi.describe())
print(boomi.info())


# Remove hidden spaces or characters explicitly
boomi['Date'] = boomi['Date'].astype(str).str.strip()

# Convert to datetime explicitly specifying dayfirst=True
boomi['Date'] = pd.to_datetime(boomi['Date'], dayfirst=True, errors='coerce')

# Extract Month and Year correctly
boomi['Month'] = boomi['Date'].dt.month
boomi['Year'] = boomi['Date'].dt.year

# Display problematic rows if any
if boomi['Date'].isna().any():
    print("Problematic rows:")
    print(boomi[boomi['Date'].isna()])
else:
    print(boomi.head())

# Logic to create 'PRD/NON-PRD' column based on 'TenantId'
boomi['PRD/NON-PRD'] = boomi['TenantId'].apply(lambda x: 'NON-PROD' if 'nonprd' in x.lower() else ('PROD' if 'prd' in x.lower() else 'UNKNOWN'))

# Verify the DataFrame with the new columns
print(boomi.head())

# Add 'Type' column with constant value 'iPack'
boomi['Type'] = 'iPack'

# Verify the DataFrame with the new columns
print(boomi.head())


# List of all possible features
feature_list = [
    'HoursAndDollarsAllocation', 'ScheduleTemplateUpload', 'ProratedAccruals',
    'TransferReportToSFTP', 'PendingCorrectionsComputation', 'EnhancedStaffingDashboard',
    'ReligiousHolidayAdjustments', 'OpenShiftPayIncentive', 'PressGaneyNDNQIReport',
    'BrazilComplianceReports', 'FlexibleBreakAdjustment', 'MealPenalty',
    'SplitShiftPremiumandSpreadofHoursPay', 'PayPeriodMinimumWageTopUp',
    'DynamicPayCodeAllocationForActivities', 'SchedulePostAuditReport', 'MexicoXOvertime',
    'Mondayisation', 'ChileAuditReports', 'CWFMPeopleImport', 'CWFMImportExport', 'CWFMReports',
    'UltiProPeopleImport', 'JoursdeFractionnement', 'BetterOffOverallTest', 'MultiWeekWorkloadPattern',
    'LeaveLoading', 'MobilePunchTimeZoneAdjustment', 'GRIAWorkingDaysandWeekendsCompliance',
    'WorkingTimeDirectiveReport', 'SicknessPeriodManagement', 'AnonymizationTool',
    'AverageAbsenceCalculation', 'AbsenceBasedAccrualAdjustment', 'FranceVacationCalculation',
    'RightToRest', 'BrazilRoundingRules', 'FLSAiOvertimeExemption', 'GPTW', 'NYCOTCap',
    'OptimizedModulation', 'MonthlyVacationGrants', 'FinlandAccrualsCalculation',
    'PaidSickYearlyBonusPayout', 'EvidencePunchImport', 'EvidencePunchAttestation',
    'TimecardApproval', 'GTOR_Validations', 'GTORDocumentAttachment', 'TimecardValidation',
    'VolunteerToWork', 'AccrualPayoutRequest', 'ModifiedAccrualPayoutRequest', 'LeaveTimeRequest',
    'OpenShift', 'TotalizerExtensibilitySetup', 'DirectAccrualDonationv.',
    'DirectAccrualDonationCustomGroupv.', 'AccrualPool', 'ScheduleChangeAttestation',
    'PredictiveSchedulingAttestation', 'DirectAccrualDonationv', 'EmployeeManagedWorkLocations',
    'AutoShifts', 'DocumentAttachment', 'ShiftSwapWorkruleOverride', 'ShiftSwapFramework',
    'ShiftSwapWorkruleOverrideMainProcess', 'SchedulePostAcknowledgement',
    'MultipleReviewersSchedulePost', 'AccrualPool-Received', 'ReturnBasedVacationDeduction',
    'ChilePunchReceipt', 'AbsenceManagementv', 'DirectAccrualDonation', 'TimeOff',
    'GPTW_ProWFM', 'SalariedPresenceRecording', 'VolunteerToWork', 'AccrualPool'
]

# Matching logic function
def match_feature(name, feature_list):
    name_processed = name.replace(" ", "").lower()
    features_processed = [feature.lower() for feature in feature_list]
    matches = get_close_matches(name_processed, features_processed, n=1, cutoff=0.5)
    return feature_list[features_processed.index(matches[0])] if matches else 'Unknown'

# Add 'Feature' column based on fuzzy matching logic
boomi['Feature'] = boomi['Name'].apply(lambda x: match_feature(x, feature_list))

# Verify the DataFrame with the new columns
print(boomi.head())
print(boomi.info())


# Check distinct values for column 'Feature'
print("Distinct values in 'Feature':", boomi['Feature'].unique())


# Add 'Data Extract date' column with constant value 'Oct-2025'
boomi['Data Extract date'] = 'Oct-2025'

# Final verification of DataFrame---9 columns
print(boomi.head())
print(boomi.info())



################################Creating final data set#####################

# Merge workflow and boomi DataFrames row-wise
merged_df = pd.concat([workflow, boomi], ignore_index=True)

# Verify merged DataFrame dimensions
print("Merged DataFrame dimensions:", merged_df.shape)

# Export merged DataFrame to CSV
merged_df.to_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Extenson adoption data - Oct 2025\Merged_Workflow_Boomi_Oct2025.csv', index=False)

print("Merged DataFrame saved successfully!")



###################Merging with previous extract ##########################

# Read previously merged data (Apr 2025) from Excel file
aprJul2025_df = pd.read_excel(r'C:\Abhinav B\Product management\Extensions Adoption Data\Merged Usage Data For Dashboard\Merged_Workflow_Boomi_Jul2025_Apr2025.xlsx')
print(aprJul2025_df.head())


# Correct format explicitly for 'Data Extract date' in Apr 2025 data
aprJul2025_df['Data Extract date'] = pd.to_datetime(aprJul2025_df['Data Extract date']).dt.strftime('%b-%Y')



print(aprJul2025_df.head())
print(aprJul2025_df.info())


# Merge Oct 2025 and AprJul 2025 DataFrames row-wise
OctJulApr2025 = pd.concat([merged_df, aprJul2025_df], ignore_index=True)

# Verify merged DataFrame dimensions
print("Oct Jul Apr DataFrame dimensions:", OctJulApr2025.shape)

# Export the final merged DataFrame to CSV
OctJulApr2025.to_csv(r'C:\Abhinav B\Product management\Extensions Adoption Data\Merged Usage Data For Dashboard\Merged_Workflow_Boomi_OctJulApr.csv', index=False)
OctJulApr2025.to_excel(r'C:\Abhinav B\Product management\Extensions Adoption Data\Merged Usage Data For Dashboard\Merged_Workflow_Boomi_OctJulApr.xlsx', index=False)

print("aprjuloct2025 DataFrame saved successfully!")


















