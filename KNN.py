# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:57:54 2021

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
import warnings
warnings.filterwarnings("ignore")
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def getupperlower_outlier(col):
    sorted(col)
    if col.isnull().values.any()==True:
        Q1,Q3=np.nanpercentile(col,[25,75])
        print("There are NaN values in the column,which we will ignore")
    else:
        Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    print("Inside function:Lower range = ",lower_range)
    print("Inside function:Upper range = ",upper_range)
    return lower_range, upper_range



def treat_outlier_5_95(x):
    # taking 5,25,75 percentile of column
    q5= np.percentile(x,5)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    dt=np.percentile(x,95)
    #calculationg IQR range
    IQR=q75-q25
    print("INSIDE FUNCTION :Interquartile range of the column is ",IQR)
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
    print("INSIDE FUNCTION:Interquartile range of the column is ",IQR)
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
           
    
def get_num_cat_colnames(dframe):
        hl_cat=[]
        hl_num=[]
        for i in dframe.columns :
            if dframe[i].dtype=="object":
                hl_cat.append(i)
            else :
                hl_num.append(i)
        print("Inside function :Categorical cols are =",hl_cat)
        print("Inside function :Numerical cols are =",hl_num)
        return hl_cat,hl_num
    
    
 ##zero variance or constant features
def return_constant_columns(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    novar_columns = dataframe.columns[dataframe.nunique()<=1]
    print("no Variance columns are ",novar_columns)
    a=dataframe.loc[:,novar_columns]
    print(a.shape)
    return a


def select_badvalues_rows(dframe,cat_colnames,bad_vals):
    smalldfs=[]
    for i in cat_colnames :           
        smalldfs.append(dframe.loc[dframe[i].str.contains('|'.join(bad_values))==True])
        print("For column name =",i)
    print(type(smalldfs))
    largedf=pd.concat(smalldfs,ignore_index=True)    
    return largedf

# c=select_badvalues_rows(df1,cat_col,bad_values)
# cat_col,num_col=get_num_cat_colnames(df1)
    

# Use 3 decimal places in output display
pd.set_option("display.precision", 3)

# Don't wrap repr(DataFrame) across additional lines
##pd.set_option("display.expand_frame_repr", False)

# Set max rows displayed in output
pd.set_option("display.max_rows", 100)

##Set max columns displayed in output 
pd.set_option("display.max_columns", 30)

pd.set_option("display.max_colwidth",30)
##Imputing badvalues 
# df['col1'].replace('?',np.NaN,inplace=True)
# df['col1']=df['col1'].astype('float64')
# ##check and replace with col3 if condition true else let it be col2
# df['col2']=np.where(df['col2']==-1,df['col3'],df['col2'])
# ##impute with mean when bad val -1
# df['col5'].replace(-1,df['col5'].mean(),inplace=True)
# ##impute with mean/median for missing values (if no outliers then safe)
# df.col6.fillna(df.col6.mean(),inplace=True)
# from sklearn.impute import SimpleImputer 
# imputer=SimpleImputer(missing_values=np.nan,strategy='median')
# df_numericcols=pd.DataFrame(imputer.fit_transform(df_numericcols),columns=df_numericcols.columns)

##Basic EDA ::Columns dtypes,shape,unique values 
##Basic EDA ::Which Features are Categorical (Character)Or Categorical (Numerical),Which are continuous
##correlated columns,same value or Zero variance columns;; DOM -SMET (Duplicates,outliers,missing-Scaling mlticollinr,encoding,transformation)

iris_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 17-KNN NB_XGB\iris-1 (1).csv')
iris=iris_orig.copy()
print(iris.shape)

##Droping all rows with NA/NULL values
iris=iris.dropna(how='all')


print(iris.shape)
print(iris.head(5))
print(iris.dtypes)
print(iris.describe())
print(iris.info())


iris.isnull().sum()

for k in iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]:
    mean = iris[k].mean()
    print(mean)
    print("No. Of NULL values in this column are",iris[k].isnull().sum())
    iris[k].fillna(mean,inplace=True)
##Duplicate analysis 
dups=iris.duplicated()
print("The num of duplicate records are =",dups.sum())

##encoding 
iris["Species"]=pd.Categorical(iris["Species"]).codes

##variance 
iris.var()


##train and validate split
all_labels=iris['Species']
all_ind_ds=iris.drop(["Species"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.80, random_state=20)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

iris['Species'].value_counts()
iris.boxplot()
plt.xticks(rotation=90)


from sklearn.neighbors import KNeighborsClassifier
# initiantiate learning model (k = 3)
KNN_model=KNeighborsClassifier(n_neighbors = 3,metric='euclidean')

# fitting the model
KNN_model.fit(trainset,train_labels)

# predict the response
pred_test = KNN_model.predict(testset)

# evaluate accuracy
print("Accuracy Score for K=3 is ", accuracy_score(test_labels, pred_test))

# initiantiate learning model (k = 5)
KNN_model=KNeighborsClassifier(n_neighbors = 5,metric='euclidean')

# fitting the model
KNN_model.fit(trainset,train_labels)

# predict the response
pred_test_5 = KNN_model.predict(testset)

# evaluate accuracy
print("Accuracy Score for K=5 is ", accuracy_score(test_labels,pred_test_5))

# initiantiate learning model (k = 9)
from sklearn.neighbors import KNeighborsClassifier
KNN_model=KNeighborsClassifier(n_neighbors = 9,metric='euclidean')

# fitting the model
KNN_model.fit(trainset,train_labels)

# predict the response
pred_test = KNN_model.predict(testset)

# evaluate accuracy
print("Accuracy Score for K=9 is ", accuracy_score(test_labels, pred_test))


# ##optimal K
# Question 9 - Optimal no. of K
# Run the KNN with no of neighbours to be 1,3,5..19 and *Find the optimal number of neighbours from the above list using the Mis classification error
# Hint:# Misclassification error (MCE) = 1 - Test accuracy score. Calculated MCE for each model with neighbours = 1,3,5...19 and find the model with lowest MCE

# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainset, train_labels)
    # evaluate accuracy
    scores = knn.score(testset, test_labels)
    ac_scores.append(scores)

# changing to misclassification error
MCE = [1 - x for x in ac_scores]
MCE



# plot misclassification error vs k
plt.plot(range(1,20,2), MCE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


##Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_obj=GaussianNB()
nb_model=nb_obj.fit(trainset,train_labels)

pred_train=nb_model.predict(trainset)
print("accuracy on trainset",accuracy_score(train_labels,pred_train))

pred_test=nb_model.predict(testset)
print("accuracy on testset",accuracy_score(test_labels,pred_test))

print("Confusion Matrix")
cm=metrics.confusion_matrix(test_labels, pred_test, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)


######################quiz
k_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 17-KNN NB_XGB\default_2k.csv')
k=k_orig.copy()
print(k.shape)

##Droping all rows with NA/NULL values
k=k.dropna(how='all')


print(k.shape)
print(k.head(5))
print(k.dtypes)
print(k.describe())
print(k.info())



for j in k[['student','default']]: 
      k[j] = pd.Categorical(k[j]).codes 
      print('Done for Column :',j)
      
 ##split
##train and validate split
all_labels=k['default']
all_ind_ds=k.drop(["default"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=1)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)


nb_obj=GaussianNB()
nb_model=nb_obj.fit(trainset,train_labels)

pred_train=nb_model.predict(trainset)
print("accuracy on trainset",accuracy_score(train_labels,pred_train))

pred_test=nb_model.predict(testset)
print("accuracy on testset",accuracy_score(test_labels,pred_test))
classification_report(test_labels,pred_test)
recall_score(test_labels,pred_test)
confusion_matrix(test_labels,pred_test)

cols=['DEFAULT_Yes','STUD_Yes']

dummies=pd.get_dummies(k[["default", "student"]], columns=["default", "student"], prefix=["DEFAULT","STUD"],drop_first=True)

k= pd.concat([k, dummies], axis=1)
k.drop(cols, axis = 1, inplace=True)
k.head()

        
dups=k.duplicated()
print("The num of duplicate records are =",dups.sum())  

k.boxplot()
plt.xticks(rotation=90)     
        
k.isnull().sum()
k.info()        

##Multicollinearity 
sns.pairplot(k)
plt.show()
##Heatmap with masking 
plt.figure(figsize=(12,7))
sns.heatmap(k.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(k.corr()) )
plt.show()

##split
##train and validate split
all_labels=k['default']
all_ind_ds=k.drop(["default"],axis=1)
# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=999)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

from scipy.stats import zscore



sobj=StandardScaler()
data_scaled_trainset=sobj.fit_transform(trainset)
data_scaled_testset=sobj.transform(testset)        
        
train_scaled=trainset.apply(zscore)        
test_scaled=testset.apply(zscore)       

NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )
NNH=NNH.fit(trainset,train_labels)
pred_train=NNH.predict(trainset)
accuracy_score(train_labels,pred_train)

pred_test=NNH.predict(testset)
accuracy_score(test_labels,pred_test)

confusion_matrix(test_labels,pred_test)


# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainset, train_labels)
    # evaluate accuracy
    scores = knn.score(testset, test_labels)
    ac_scores.append(scores)

# changing to misclassification error
MCE = [1 - x for x in ac_scores]
MCE



# plot misclassification error vs k
plt.plot(range(1,20,2), MCE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
