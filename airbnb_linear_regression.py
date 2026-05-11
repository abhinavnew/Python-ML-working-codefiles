# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:04:46 2021

@author: Abhinav.Bajpai
"""

import gc
##Clear variable/objects from workspace to free up memory
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
dir()
import numpy as np 
print("numpy path is",np.__path__)
print('The numpy version is {}.'.format(np.__version__))

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
print('The pandas version is {}.'.format(pd.__version__))
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics ##for rmse
import statsmodels.formula.api as smf ##LinReg using statsmodel


def getupperlower_outlier(col):
    sorted(col)
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
def drop_constant_columns_4(dataframe):
    """
    Drops constant value columns of pandas dataframe.
    """
    print(dataframe.shape)
    keep_columns = dataframe.columns[dataframe.nunique()>1]
    a=dataframe.loc[:,keep_columns].copy()
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
pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth",20)
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

airbnb_orig =pd.read_csv(r'C:\Abhinav B\Kaggle\Python related\DSBA related\Week 14 _linear regression\AirBNB.csv')
airbnb=airbnb_orig.copy()


print(airbnb.shape)
print(airbnb.head(5))
print(airbnb.dtypes)
print(airbnb.describe())
print(airbnb.info())

##Categorical variables -univariate
print(airbnb['room_type'].unique())
print(airbnb['cancellation_policy'].unique())
print(airbnb['cleaning_fee'].unique())
print(airbnb['instant_bookable'].unique())
airbnb['room_type'].value_counts().plot(kind='bar')
airbnb['cancellation_policy'].value_counts().plot(kind='bar')
airbnb['cleaning_fee'].value_counts().plot(kind='bar')
airbnb['instant_bookable'].value_counts().plot(kind='bar')
##Getting value counts of cat variables
for column in airbnb.columns:
    if airbnb[column].dtype == 'object':
        print(column.upper(),': ',airbnb[column].nunique())
        print(airbnb[column].value_counts().sort_values())
        print('\n')

##Continuous -Unique values
print(airbnb[['accommodates','bathrooms','review_scores_rating','bedrooms','beds']].describe())
print(airbnb['accommodates'].unique())
print(airbnb['bathrooms'].unique())
print(airbnb['review_scores_rating'].unique())
print(airbnb['bedrooms'].unique())
print(airbnb['beds'].unique())
##Discrete numerical -value counts and histogram for distribution
sns.distplot(airbnb.accommodates,bins=10)
airbnb['accommodates'].value_counts().plot(kind='bar')
sns.distplot(airbnb.bathrooms,bins=10)
airbnb['bathrooms'].value_counts().plot(kind='bar')
sns.distplot(airbnb.bedrooms,bins=10)
airbnb['bedrooms'].value_counts().plot(kind='bar')
sns.distplot(airbnb.beds,bins=10)
airbnb['beds'].value_counts().plot(kind='bar')
sns.distplot(airbnb.review_scores_rating,bins=10)
airbnb['review_scores_rating'].value_counts().plot(kind='bar')

##Checking skewness for symmetry 
print(airbnb['accommodates'].skew())
print(airbnb['bathrooms'].skew())
print(airbnb['bedrooms'].skew())
print(airbnb['beds'].skew())
print(airbnb['review_scores_rating'].skew())


##target variables distribution 
sns.distplot(airbnb.log_price,bins=10)

##remove Duplciates

dups=airbnb.duplicated()
print("The num of duplicate records are =",dups.sum())
airbnb[dups]
##No duplicates found 

airbnb.describe(include="all")

##Missing values-need to be imputed first otherwise outlier treatment not possible
airbnb.isnull().sum()
airbnb.info()
##Imputing numerical/continuous numerical/discrete cols with median
for j in airbnb.columns:
    if airbnb[j].dtype != 'object':
        median = airbnb[j].median()
        airbnb[j] = airbnb[j].fillna(median)   

print("after imputation of numerical columns ",airbnb.isnull().sum())

##one hot for categorical variables with no order 

airbnb = pd.get_dummies(airbnb, columns=['room_type','cancellation_policy','cleaning_fee'],drop_first=True)
print("after OneHot of categorical columns ",airbnb.isnull().sum())

##Outliers
airbnb.boxplot()
plt.xticks(rotation=90);
sns.boxplot(airbnb['accommodates'])
sns.boxplot(airbnb['bathrooms'])
sns.boxplot(airbnb['beds'])

ll1,ul1=getupperlower_outlier(airbnb['accommodates'])
out_accommodates=airbnb.loc[(airbnb['accommodates']>ul1)|(airbnb['accommodates']<ll1),]
print(out_accommodates)

ll2,ul2=getupperlower_outlier(airbnb['bathrooms'])
out_bathrooms=airbnb.loc[(airbnb['bathrooms']>ul2)|(airbnb['bathrooms']<ll2),]
print(out_bathrooms)

cont=airbnb.dtypes[(airbnb.dtypes!='uint8') & (airbnb.dtypes!='bool')].index
plt.figure(figsize=(10,10))
airbnb[cont].boxplot(vert=0)
plt.title('With Outliers',fontsize=16)
plt.show()

##Outlier treatement with ul ll winsorization 
cat_col,num_col=get_num_cat_colnames(airbnb)

airbnb.boxplot()
plt.xticks(rotation=90);

for k in num_col:
     airbnb[k]=treat_outlier_ul_ll_winsor(airbnb[k])
     
airbnb.boxplot()
plt.xticks(rotation=90);

##Multicollinearity 
plt.figure(figsize=(12,7))
sns.heatmap(airbnb.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()

##Drop id column 
airbnb = airbnb.drop('id', axis=1)
airbnb.instant_bookable.replace(['f','t'],[False,True],inplace=True )


##train test split 
# Copy all the predictor variables into X dataframe
X = airbnb.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = airbnb[['log_price']]

# Split X and y into training and test set in 75:25 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=1)

rm_sk = LinearRegression()
rm_sk.fit(X_train, y_train)


rm_sk.coef_
rm_sk.coef_[0]
for idx, col_name in enumerate(X_train.columns):
    print(rm_sk.coef_[0][idx])

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, round(rm_sk.coef_[0][idx],3)))
        
intercept = rm_sk.intercept_[0]
print("The intercept for our model is {}".format(intercept))
    
rm_sk.score(X_train, y_train)
rm_sk.score(X_test, y_test)

#RMSE on Testing data
predicted_test=rm_sk.fit(X_train, y_train).predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test,predicted_test))

##Linear Regression using statsmodels
# concatenate X and y into a single dataframe
data_train = pd.concat([X_train, y_train], axis=1)
data_test=pd.concat([X_test,y_test],axis=1)
data_train.head()

data_train.rename(columns = {"room_type_Entire home/apt": "room_type_entire_home", "room_type_Private room": "room_type_private_room", 
                     "room_type_Shared room": "room_type_shared_room"}, 
                      inplace = True) 

data_test.rename(columns = {"room_type_Entire home/apt": "room_type_entire_home", "room_type_Private room": "room_type_private_room", 
                     "room_type_Shared room": "room_type_shared_room"}, 
                      inplace = True) 


expr= 'log_price ~ accommodates + bathrooms + instant_bookable+review_scores_rating +bedrooms + beds + room_type_private_room + room_type_shared_room  + cancellation_policy_moderate + cancellation_policy_strict + cleaning_fee_True'

lm1 = smf.ols(formula= expr, data = data_train).fit()
lm1.params
print(lm1.summary())

# Calculate MSE
mse = np.mean((lm1.predict(data_train.drop('log_price',axis=1))-data_train['log_price'])**2)
#Root Mean Squared Error - RMSE
np.sqrt(mse)
# Prediction on Test data
y_pred = lm1.predict(data_test)
plt.scatter(y_test['log_price'], y_pred)
plt.show()
for i,j in np.array(lm1.params.reset_index()):
    print('({}) * {} +'.format(round(j,2),i),end=' ')

####quiz


three_orig =pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Week 14 _linear regression\ThreeCars.csv')
three=three_orig.copy()
three.drop(three.columns[three.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
X = three.drop('Price', axis=1)

# Copy target into the y dataframe. 
y = three[['Price']]

# Split X and y into training and test set in 75:25 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=1)

rm_sk1 = LinearRegression()
rm_sk1.fit(X, y)


rm_sk.coef_
rm_sk.coef_[0]
for idx, col_name in enumerate(X_train.columns):
    print(rm_sk1.coef_[0][idx])
    
    for idx, col_name in enumerate(X.columns):
      print("The coefficient for {} is {}".format(col_name, round(rm_sk1.coef_[0][idx],3)))

intercept = rm_sk1.intercept_[0]
print("The intercept for our model is {}".format(intercept))

rm_sk1.score(X ,y)










