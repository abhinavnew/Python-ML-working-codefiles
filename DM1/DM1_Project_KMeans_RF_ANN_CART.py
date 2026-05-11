# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 08:58:40 2021

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
from sklearn.cluster import KMeans  ## for k means
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
sns.set(color_codes=True) 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.neural_network import MLPClassifier


def getupperlower_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


def treat_outlier_5_95(x):
    # taking 5,25,75 percentile of column
    q5= np.percentile(x,5)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    dt=np.percentile(x,95)
    #calculationg IQR range
    IQR=q75-q25
    print("Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: dt if y > upper_bound else y).apply(lambda y: q5 if y < lower_bound else y)



def treat_outlier_ul_ll_winsor(x):
    # taking 5,25,75 percentile of column
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
   #calculationg IQR range
    IQR=q75-q25
    print("Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: upper_bound if y > upper_bound else y).apply(lambda y: lower_bound if y < lower_bound else y)




def missvalue_replacenull(dframe,cat_colnames):
    for i in cat_colnames :
        a=dframe[dframe[i].str.contains('\?') | dframe[i].str.contains("NA") | dframe[i].str.contains("NaN")]
        print("records containing ? NA NAN etc are  ",a.shape )
        print("For column name =",i)
        idx=a.index
        

    
def get_num_cat_colnames(dframe):
        hl_cat=[]
        hl_num=[]
        for i in dframe.columns :
            if dframe[i].dtype=="object":
                hl_cat.append(i)
            else :
                hl_num.append(i)
        print(hl_cat)
        print(hl_num)
        return hl_cat,hl_num
    
    
 ##zero variance or constant features
def drop_constant_columns_4(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    keep_columns = dataframe.columns[dataframe.nunique()>1]
    a=dataframe.loc[:,keep_columns].copy()
    print(a.shape)
    return a
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)

##Basic EDA ::Columns dtypes,shape,unique values 
##Basic EDA ::Which Features are Categorical (Character)Or Categorical (Numerical),Which are continuous
##correlated columns,same value or Zero variance columns;; DOM -SMET (Duplicates,outliers,missing-Scaling mlticollinr,encoding,transformation)


bankmark_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\Kmeans_CART_RF_ANN_project\bank_marketing_part1_Data.csv')
bankmark=bankmark_orig.copy()

bankmark.shape
bankmark.columns
bankmark.info()
bankmark.head(10)

# Problem 1: Clustering

# A leading bank wants to develop a customer segmentation to give promotional offers to its customers. 
# They collected a sample that summarizes the activities of users during the past few months. 
# You are given the task to identify the segments based on credit card usage.

# 1.1 Read the data, do the necessary initial steps, and exploratory data analysis 
# (Univariate, Bi-variate, and multivariate analysis).


#Duplicates 
dups=bankmark.duplicated()
print("the no of duplicates =",dups.sum())
##Check for any negative values in any column
bankmark.columns[(bankmark<0).any()]

##Outliers
##Bring values to actual scale 
bankmark['spending']=bankmark['spending']*1000
bankmark['advance_payments']=bankmark['advance_payments']*100
bankmark['current_balance']=bankmark['current_balance']*1000
bankmark['credit_limit']=bankmark['credit_limit']*1000
bankmark['min_payment_amt']=bankmark['min_payment_amt']*100
bankmark['max_spent_in_single_shopping']=bankmark['max_spent_in_single_shopping']*1000

bankmark.head(10)

##Outliers
bankmark.boxplot()
plt.xticks(rotation=90);
##Outliers seen on min_payment_amt 
##Outlier records
ll1,ul1=getupperlower_outlier(bankmark['min_payment_amt'])
print(ll1,ul1)
out_minpayment=bankmark.loc[(bankmark['min_payment_amt']>ul1) | (bankmark['min_payment_amt']<ll1),]
print(out_minpayment)

ll2,ul2=getupperlower_outlier(bankmark['probability_of_full_payment'])
print(ll2,ul2)
out_prob_fullpayment=bankmark.loc[(bankmark['probability_of_full_payment']>ul2) | (bankmark['probability_of_full_payment']<ll2),]
print(out_prob_fullpayment)
##Outlier treatement -as they will disturb clustering
bankmark['min_payment_amt']=treat_outlier_ul_ll_winsor(bankmark['min_payment_amt'])
bankmark['probability_of_full_payment']=treat_outlier_ul_ll_winsor(bankmark['probability_of_full_payment'])

out_minpayment_records_after=bankmark.loc[(bankmark['min_payment_amt']>ul1),]
print(out_minpayment_records_after)

out_prob_fullpayment_after=bankmark.loc[(bankmark['probability_of_full_payment']>ul2) | (bankmark['probability_of_full_payment']<ll2),]
print(out_prob_fullpayment_after)

bankmark.boxplot()
plt.xticks(rotation=90);


##Missing values
bankmark.isnull().sum()
bankmark.info()
##No missing values found 

##Univariate analysis 
sns.distplot(bankmark.spending,bins=10)
print(bankmark['spending'].skew())
sns.distplot(bankmark.advance_payments,bins=10)
print(bankmark['advance_payments'].skew())
sns.distplot(bankmark.current_balance,bins=8)
print(bankmark['current_balance'].skew())
sns.distplot(bankmark.credit_limit,bins=10)
print(bankmark['credit_limit'].skew())
sns.distplot(bankmark.min_payment_amt,bins=10)
print(bankmark['min_payment_amt'].skew())
sns.distplot(bankmark.max_spent_in_single_shopping,bins=10)
print(bankmark['max_spent_in_single_shopping'].skew())
sns.distplot(bankmark.max_spent_in_single_shopping,bins=10)
print(bankmark['max_spent_in_single_shopping'].skew())
sns.distplot(bankmark.probability_of_full_payment,bins=10)
print(bankmark['probability_of_full_payment'].skew())


##Bivariate analysis 
sns.pairplot(bankmark)
plt.show()

##Spending ~ ad_payment ;current_balance ;credit_limit;max_spent_in_single_shopping are highly correlated
##many highly correlated ind variables are seen 

plt.figure(figsize=(12,7))
sns.heatmap(bankmark.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()


##Scale then PCA;Hinton;profile plot for each variable 

##1.2  Do you think scaling is necessary for clustering in this case? Justify

##Scaling is necessary for clustering as the many id variables like payments,max spent are much smaller than
##mention Euclidean distance 
##We do not want our algorithm to be affected by the magnitude of these variables. 
#The algorithm should not be biased towards variables with higher magnitude


##Scaling 
sobj=StandardScaler()
data_scaled_bankmark=sobj.fit_transform(bankmark)
bankmark_scaled=pd.DataFrame(data_scaled_bankmark,columns=bankmark.columns)

bankmark_scaled.head(10)
bankmark_scaled.boxplot()
plt.xticks(rotation=90);


# 1.3 Apply hierarchical clustering to scaled data. Identify the number of optimum clusters 
##using Dendrogram and briefly describe them

wardlink = linkage(bankmark_scaled, method = 'ward')
dend = dendrogram(wardlink,
                 truncate_mode='lastp',
                 p = 10,
                 )


# The clades are arranged according to how similar (or dissimilar) they are. 
# Clades that are close to the same height are similar to each other; 
# clades with different heights are dissimilar — the greater the difference in height,
#  the more dissimilarity (you can measure similarity in many different ways
##Hence 3 chosen 

hclusters=fcluster(wardlink,3,criterion='maxclust')
agg=pd.DataFrame(hclusters)

bankmark_scaled['h_cluster']=hclusters
##Clusters per spending and payment probability 
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(bankmark_scaled['spending'],bankmark_scaled['probability_of_full_payment'],
                     c=agg[0],s=50)
ax.set_title('Agglomerative Clustering')
ax.set_xlabel('Spending')
ax.set_ylabel('Probability of Payment')
plt.colorbar(scatter)

##1.4 Apply K-Means clustering on scaled data and determine optimum clusters. 
##Apply elbow curve and silhouette score. Explain the results properly. 
##Interpret and write inferences on the finalized clusters.

wss=[]
for i in range(1,11):
    KM = KMeans(n_clusters=i)
    KM.fit(bankmark_scaled)
    wss.append(KM.inertia_)

plt.plot(range(1,11), wss)
##As per the scree plot ,3 is the optimal no of clusters as after 
##that the fall in within sum of squares is not so much 

k_means = KMeans(n_clusters = 3)
k_means.fit(bankmark_scaled)
labels3 = k_means.labels_
bankmark_scaled['KM_Cluster']=labels3
##Silhoutte score 
print("k=3 sil score is ",silhouette_score(bankmark_scaled,labels3))

k_means = KMeans(n_clusters = 2)
k_means.fit(bankmark_scaled)
labels2 = k_means.labels_
##bankmark_scaled['KM_Cluster']=labels
##Silhoutte score 
print("k=2 sil score is ",silhouette_score(bankmark_scaled,labels2))

k_means = KMeans(n_clusters = 4)
k_means.fit(bankmark_scaled)
labels4 = k_means.labels_
#bankmark_scaled['KM_Cluster']=labels
##Silhoutte score 
print("k=4 sil score is ",silhouette_score(bankmark_scaled,labels4))

k_means = KMeans(n_clusters = 5)
k_means.fit(bankmark_scaled)
labels5 = k_means.labels_
#bankmark_scaled['KM_Cluster']=labels
##Silhoutte score 
print("k=5 sil score is ",silhouette_score(bankmark_scaled,labels5))


##s_width=silhouette_samples(bankmark_scaled,labels)



##1.5 Describe cluster profiles for the clusters defined. 
##Recommend different promotional strategies for different clusters.

##Profiling of clusters 
bankmark['km_cluster']=labels3

km=pd.DataFrame(labels3)

##Clusters wrt Spending and Probability of payment 
fig = plt.figure()
ax1 = fig.add_subplot(111)
scatter = ax1.scatter(bankmark_scaled['spending'],bankmark_scaled['probability_of_full_payment'],
                     c=km[0],s=50)
ax1.set_title('K-Means Clustering')
ax1.set_xlabel('Spending')
ax1.set_ylabel('Probability of Payment')
plt.colorbar(scatter)

##Clusters wrt credit_limit and Probability of payment 
fig = plt.figure()
ax2 = fig.add_subplot(111)
scatter = ax2.scatter(bankmark_scaled['credit_limit'],bankmark_scaled['probability_of_full_payment'],
                     c=km[0],s=50)
ax2.set_title('K-Means Clustering')
ax2.set_xlabel('Credit LIMIT')
ax2.set_ylabel('Probability of Payment')
plt.colorbar(scatter)


##Clusters wrt credit_limit and Probability of payment 
fig = plt.figure()
ax3 = fig.add_subplot(111)
scatter = ax3.scatter(bankmark_scaled['max_spent_in_single_shopping'],bankmark_scaled['probability_of_full_payment'],
                     c=km[0],s=50)
ax3.set_title('K-Means Clustering')
ax3.set_xlabel('Max Spent In Single Shopping (impulsiveness)')
ax3.set_ylabel('Probability of Payment')
plt.colorbar(scatter)




bankmark.km_cluster.value_counts().sort_index()

clust_profile=bankmark
clust_profile=clust_profile.groupby('km_cluster').mean()
clust_profile['freq']=bankmark['km_cluster'].value_counts().sort_index()
print(clust_profile)
## Cluster 0 represents Low mean credit limit customers probably newbies with a low probability of full payment which means credit will be rotatated
##Such customer are profitable to the credit card company as they generate interest levied by the bank and should be targeted for more offers and spending but care should be taken to monitor txns when they are nearning their credit limit

##Cluster 1 represent high spenders(Highest mean spending)  and impulsive/big buyers as captured by max_spent_in_single_shopping 
##With the best probability of repayment of full amount ,these are good customers (spend big and repay back in full)
##bank should target them with offer ,points to maintain loyalty as they act as brand ambassadors 
##Wealthy as advance payments also made

##Cluster 2 are mid way group with medium spending and good probability of full repay
##these are potetial high spenders and should be targeted with appropraite campaign








