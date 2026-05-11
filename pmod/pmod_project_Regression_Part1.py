# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:46:33 2021

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
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def getupperlower_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    print("Interquartile range of the column {} is {} ".format(col,IQR))
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

cubic_orig=pd.read_csv(r'E:\AbhinavB\Kaggle\Python related\DSBA related\Competions\Logistic_linReg_LDA_Project\cubic_zirconia.csv')
cubic=cubic_orig.copy()
cubic.drop(cubic.columns[cubic.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


print(cubic.shape)
print(cubic.head(5))
print(cubic.dtypes)
print(cubic.describe())
print(cubic.info())

##Univariate analysis -Categorical
print(cubic['cut'].unique())
print(cubic['color'].unique())
print(cubic['clarity'].unique())

#Count bar plots for categorical variables 
cubic['cut'].value_counts().plot(kind='bar')

cubic['color'].value_counts().plot(kind='bar')

cubic['clarity'].value_counts().plot(kind='bar')

#Histograms for continuous variables
sns.distplot(cubic.carat,bins=10)
sns.distplot(cubic.depth,bins=10)
sns.distplot(cubic.table,bins=10)
sns.distplot(cubic.x,bins=10)
sns.distplot(cubic.y,bins=10)
sns.distplot(cubic.z,bins=10)
sns.distplot(cubic.price,bins=10)
##stats
cubic['price'].describe()
##Checking skewness for symmetry 
print(cubic['carat'].skew())
print(cubic['depth'].skew())
print(cubic['table'].skew())
print(cubic['x'].skew())
print(cubic['y'].skew())
print(cubic['z'].skew())
print(cubic['price'].skew())


##Bivariate analysis 
sns.pairplot(cubic)
plt.show()

##Heatmap with masking 

plt.figure(figsize=(12,7))
sns.heatmap(cubic.corr(),annot=True,fmt='.2f',cmap='rainbow',mask=np.triu(cubic.corr()) )
plt.show()


##Duplicates 
dups=cubic.duplicated()
print("The num of duplicate records are =",dups.sum())
cubic[dups]
cubic.drop_duplicates(inplace=True)
print(cubic.shape)

##outlier analysis
cubic.boxplot()
plt.xticks(rotation=90)

ll1,ul1=getupperlower_outlier(cubic['carat'])
out1=cubic.loc[(cubic['carat']>ul1)|(cubic['carat']<ll1),]
print(out1)

ll2,ul2=getupperlower_outlier(cubic['depth'])
out2=cubic.loc[(cubic['depth']>ul2)|(cubic['depth']<ll2),]
print(out2)

ll3,ul3=getupperlower_outlier(cubic['table'])
out3=cubic.loc[(cubic['table']>ul3)|(cubic['table']<ll3),]
print(out3)

ll4,ul4=getupperlower_outlier(cubic['x'])
out4=cubic.loc[(cubic['x']>ul4)|(cubic['x']<ll4),]
print(out4)

ll5,ul5=getupperlower_outlier(cubic['y'])
out5=cubic.loc[(cubic['y']>ul5)|(cubic['y']<ll5),]
print(out5)

ll6,ul6=getupperlower_outlier(cubic['z'])
out6=cubic.loc[(cubic['z']>ul6)|(cubic['z']<ll6),]
print(out6)

##Outlier treatement -as they will disturb line of best fit
cubic['carat']=treat_outlier_ul_ll_winsor(cubic['carat'])
cubic['table']=treat_outlier_ul_ll_winsor(cubic['table'])
cubic['x']=treat_outlier_ul_ll_winsor(cubic['x'])
cubic['y']=treat_outlier_ul_ll_winsor(cubic['y'])
cubic['z']=treat_outlier_ul_ll_winsor(cubic['z'])

cubic.boxplot()
plt.xticks(rotation=90)

##Missing values /bad values 

cubic.isnull().sum()
cubic.info()
#get any bad values
bad_values = ['\?','NA','\@','None','NaN','Nan','nan','Missing','-99','-999','\??','\???'] 
cat_col,num_col=get_num_cat_colnames(cubic)
c=select_badvalues_rows(cubic,cat_col,bad_values)
print(c)

##Any constant columns
d= return_constant_columns(cubic)
print(d)

##Checking for zero values where it does not make sense

zer=cubic.loc[(cubic['x']==0) | (cubic['y']==0) | (cubic['z']==0)]
print(zer)

##Imputation with MEAN

for i in cubic[['carat', 'depth', 'table', 'x', 'y','z']]:
    median = cubic[i].mean()
    cubic[i].replace(np.nan, median, inplace= True)
    
cubic.isnull().sum()

##Scaling 
##Scaling not required for linear regression 


##Multicollinearity 
sns.pairplot(cubic)
plt.show()


plt.figure(figsize=(12,7))
sns.heatmap(cubic.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()

##Encoding categorical nominal variables to numerical values as per order-label encoding
encobj_cut=OrdinalEncoder(categories=[['Ideal' ,'Premium' ,'Very Good' ,'Good' ,'Fair']])
cubic['cut']=encobj_cut.fit_transform(cubic[["cut"]])

encobj_color=OrdinalEncoder(categories=[['D', 'E' , 'F', 'G' ,'H' ,'I' ,'J']])
cubic['color']=encobj_color.fit_transform(cubic[["color"]])

encobj_clarity=OrdinalEncoder(categories=[['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1', 'I2', 'I3']])
cubic['clarity']=encobj_clarity.fit_transform(cubic[["clarity"]])


##train test split 
# Copy all the predictor variables into X dataframe
all_ind_ds= cubic.drop('price', axis=1)
# Copy target into the y dataframe. 
all_labels = cubic[['price']]

# Split X and y into training and test set in 75:25 ratio

trainset, testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=0.30 , random_state=888)
print(trainset.shape)
print(train_labels.shape)
print(testset.shape)
print(test_labels.shape)

##Baseline model 
BaselinePricePred=test_labels.mean()
print(BaselinePricePred)
a=[BaselinePricePred]*len(test_labels)
print(np.sqrt(metrics.mean_squared_error(test_labels,a)))

##Linear regression model -sklearn library
rm_sk = LinearRegression()
rm_sk.fit(trainset, train_labels)
rm_sk.coef_
rm_sk.coef_[0]
for idx, col_name in enumerate(trainset.columns):
    print(rm_sk.coef_[0][idx])

for idx, col_name in enumerate(trainset.columns):
    print("The coefficient for {} is {}".format(col_name, round(rm_sk.coef_[0][idx],3)))
        
intercept = rm_sk.intercept_[0]
print("The intercept for our model is {}".format(intercept))
    
rm_sk.score(trainset, train_labels)
rm_sk.score(testset, test_labels)

#RMSE on Testing data
predicted_test=rm_sk.predict(testset)
predicted_train=rm_sk.predict(trainset)
np.sqrt(metrics.mean_squared_error(train_labels,predicted_train))
np.sqrt(metrics.mean_squared_error(test_labels,predicted_test))



##Other regression models -Default hyperparameters
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
dtr = tree.DecisionTreeRegressor(random_state=888)
rfr=RandomForestRegressor(random_state=888)
linreg=LinearRegression()
models=[linreg,dtr,rfr]
rmse_train=[]
rmse_test=[]
scores_train=[]
scores_test=[]

for i in models:
    i.fit(trainset,train_labels)
    scores_train.append(i.score(trainset, train_labels))
    scores_test.append(i.score(testset, test_labels))
    rmse_train.append(np.sqrt(mean_squared_error(train_labels,i.predict(trainset))))
    rmse_test.append(np.sqrt(mean_squared_error(test_labels,i.predict(testset))))

print(pd.DataFrame({'Train RMSE': rmse_train,'Test RMSE': rmse_test,'Training Score':scores_train,'Test Score': scores_test},
            index=['Linear Regression','Decision Tree Regressor','Random Forest Regressor']))


##Comparison other models with gridsearch on hyperparameters
##grid search on dtree CART
start_time1 = time.monotonic()
print(start_time1)
param_grid = {
    'criterion': ['mse','mae'],
    'max_depth': [10,15,20,25,30],
    'min_samples_leaf': [3,15,30],
    'min_samples_split': [15,30,35,40,50],
}
dtr=tree.DecisionTreeRegressor(random_state=888)
grid_search = GridSearchCV(estimator = dtr, param_grid = param_grid, cv = 3,verbose=30)
grid_search.fit(trainset,train_labels)
print(grid_search.best_params_)
##Capturing training time 
end_time1 = time.monotonic()
print(end_time1)
cart_train_time=timedelta(seconds=end_time1 - start_time1)
print("training time for CART model is ",cart_train_time)

cart_mod_bst=grid_search.best_estimator_
pred_cart_test1 = cart_mod_bst.predict(testset)
pred_cart_train1=cart_mod_bst.predict(trainset)  

##Feature importance 
x2=pd.DataFrame(cart_mod_bst.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x2[0],x2.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in %')
plt.title('Feature Importance Plot')
plt.show()



##Random Forest regressor 
start_time2 = time.monotonic()
print(start_time2)
param_grid = {
    'max_depth': [7,10],
    'max_features': [4, 6],
    'min_samples_leaf': [3, 15,30],
    'min_samples_split': [30, 50,100],
    'n_estimators': [300, 500]
}

rfr = RandomForestRegressor(random_state=123)

grid_search = GridSearchCV(estimator = rfr, param_grid = param_grid, cv = 3,verbose=30)

grid_search.fit(trainset,train_labels)

print(grid_search.best_params_)

##Capturing training time 
end_time2 = time.monotonic()
print(end_time2)
rf_train_time=timedelta(seconds=end_time2 - start_time2)
print("training time for CART model is ",rf_train_time)

rfc_mod_bst=grid_search.best_estimator_
pred_test_rf=rfc_mod_bst.predict(testset)
pred_train_rf=rfc_mod_bst.predict(trainset)

##feature importance randomForest
x55=pd.DataFrame(rfc_mod_bst.feature_importances_*100,index=trainset.columns).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,7))
sns.barplot(x55[0],x55.index,palette='rainbow')
plt.ylabel('Feature Name')
plt.xlabel('Feature Importance in % For RandomForest model')
plt.title('Feature Importance Plot')
plt.show()

##compiling tuned (by gridsearch) results

rfr2 = RandomForestRegressor(max_depth=10, max_features=6, 
                            min_samples_leaf= 3,
                            min_samples_split= 30, n_estimators= 500,
                            random_state=888)
dtr2 = tree.DecisionTreeRegressor(max_depth=20,min_samples_split=35,min_samples_leaf=3,random_state=888)
linreg2 = LinearRegression()

models=[linreg2,dtr2,rfr2]

rmse_train=[]
rmse_test=[]
scores_train=[]
scores_test=[]


for i in models:
    i.fit(trainset,train_labels)
    scores_train.append(i.score(trainset, train_labels))
    scores_test.append(i.score(testset, test_labels))
    rmse_train.append(np.sqrt(mean_squared_error(train_labels,i.predict(trainset))))
    rmse_test.append(np.sqrt(mean_squared_error(test_labels,i.predict(testset))))

print(pd.DataFrame({'Train RMSE': rmse_train,'Test RMSE': rmse_test,'Training Score':scores_train,'Test Score': scores_test},
            index=['Linear Regression','Decision Tree Regressor','Random Forest Regressor']))






