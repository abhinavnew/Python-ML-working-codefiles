# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:20:14 2021

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

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics ##for rmse
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import plot_importance
import pickle
from imblearn.over_sampling import SMOTE 

import nltk 
import string
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
##stemmer = SnowballStemmer("english")

print('The matplotlib version is {}.'.format(mp.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))

def getupperlower_outlier(col):
    sorted(col)
    if col.isnull().values.any()==True:
        Q1,Q3,Q1_0,Q99,Q5,Q95=np.nanpercentile(col,[25,75,1,99,5,95])
        count=col.isnull().sum()
        print("There are NaN values in the column,which we will ignore-No of NaNs=",count)
    else:
        Q1,Q3,Q1_0,Q99,Q5,Q95=np.percentile(col,[25,75,1,99,5,95])
    IQR=Q3-Q1
    print("Interquartile range of the column is ",IQR)
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    print("Inside function:Lower range = ",lower_range)
    print("Inside function:Upper range = ",upper_range)
    print("Inside function:1st percentile = ",Q1_0)
    print("Inside function:99th percentile = ",Q99)
    print("Inside function:5th percentile = ",Q5)
    print("Inside function:95th percentile = ",Q95)
    return lower_range, upper_range



def treat_outlier_5_95_winsor(x):
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

def treat_outlier_1_99_winsor(x):
    # taking 1,25,75,99 percentile of column
    q1=np.percentile(x,1)
    q25=np.percentile(x,25)
    q75=np.percentile(x,75)
    q99=np.percentile(x,99)
    #calculationg IQR range
    IQR=q75-q25
    print("INSIDE FUNCTION :Interquartile range of the column is ",IQR)
    #Calculating minimum threshold
    lower_bound=q25-(1.5*IQR)
    upper_bound=q75+(1.5*IQR)
    #Capping outliers
    return x.apply(lambda y: q99 if y > upper_bound else y).apply(lambda y: q1 if y < lower_bound else y)


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

import os

nltk.download('inaugural')
from nltk.corpus import inaugural
inaugural.fileids()
inaugural.raw('1941-Roosevelt.txt')
inaugural.raw('1961-Kennedy.txt')
inaugural.raw('1973-Nixon.txt')


roose=inaugural.raw('1941-Roosevelt.txt')
print(roose)
word_count=len(roose.split()) 
print(word_count) 
roose_words=roose.split()
roose_split_list=roose.split()

# char_count=0
# for i in roose_split_list:
#     char_count=char_count+len(i)
# print(char_count)    


char_count=0
for i in wtoken:
    char_count=char_count+len(i)
print(char_count) 

##Sentence count
##Tokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
import re
stoken=sent_tokenize(roose)
len(stoken)

##kennedy 
ken=inaugural.raw('1961-Kennedy.txt')
print(ken)
word_count_k=len(ken.split()) 
print(word_count_k) 
ken_words=ken.split()
all_words_ken_low=ken.lower()
all_words_ken=re.sub("[^\w\s]","",all_words_ken_low)
print("Kennedy speech after removing punctuation thats not word or spaces=",all_words_ken)
wtoken_k=word_tokenize(all_words_ken)
len(wtoken_k)
char_count=0
for i in wtoken_k:
    char_count=char_count+len(i)
print(char_count) 
stoken_k=sent_tokenize(ken)
len(stoken_k)

stop_words_nltk=nltk.corpus.stopwords.words('english') +list(string.punctuation)
wtoken_without_sw_k=[i for i in wtoken_k if not i in stop_words_nltk]
print(wtoken_without_sw_k)
len(wtoken_without_sw_k)

fdist_k=FreqDist(wtoken_without_sw_k)
fdist_kennedy=fdist_k.most_common(10)
print(fdist_kennedy)


##Word Cloud 
wc_k=' '.join(wtoken_without_sw_k)

wordcloud = WordCloud(width = 3000, height = 3000, 
                background_color ='black', 
                min_font_size = 10, random_state=100).generate(wc_k) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off")
plt.xlabel('Word Cloud')
plt.tight_layout(pad = 0) 

print("Word Cloud for kennedy Speech (after cleaning)!!")
plt.show()


##Nixon
nix=inaugural.raw('1973-Nixon.txt')
print(nix)
word_count_n=len(nix.split()) 
print(word_count_n) 
all_words_nix_low=nix.lower()
all_words_nix=re.sub("[^\w\s]","",all_words_nix_low)
print("nixon  speech after removing punctuation thats not word or spaces=",all_words_nix)
wtoken_n=word_tokenize(all_words_nix)
len(wtoken_n)
char_count=0
for i in wtoken_n:
    char_count=char_count+len(i)
print(char_count) 
stoken_n=sent_tokenize(nix)
len(stoken_n)

wtoken_without_sw_n=[i for i in wtoken_n if not i in stopwords.words()]
print(wtoken_without_sw_n)
len(wtoken_without_sw_n)

fdist_n=FreqDist(wtoken_without_sw_n)
fdist_nix=fdist_n.most_common(6)
print(fdist_nix)


wtoken_without_sw=[i for i in wtoken if not i in stopwords.words()]
print(wtoken_without_sw)

wc_n=' '.join(wtoken_without_sw_n)

wordcloud = WordCloud(width = 3000, height = 3000, 
                background_color ='black', 
                min_font_size = 10, random_state=100).generate(wc_n) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off")
plt.xlabel('Word Cloud')
plt.tight_layout(pad = 0) 

print("Word Cloud for Nixon Speech (after cleaning)!!")
plt.show()




##Stopwords
stop_words_nltk=nltk.corpus.stopwords.words('english') +list(string.punctuation)

all_words_roose_low=roose.lower()
all_words_roose=re.sub("[^\w\s]","",all_words_roose_low)
print("Roosevelt speech after removing punctuation thats not word or spaces=",all_words_roose)
wtoken=word_tokenize(all_words_roose)
len(wtoken)
wtoken_without_sw=[i for i in wtoken if not i in stopwords.words()]
print(wtoken_without_sw)


fdist=FreqDist(wtoken_without_sw)

fdist1=fdist.most_common(3)
print(fdist1)


stoken=sent_tokenize(roose)
len(stoken)

##Word Cloud 
wc_a=' '.join(wtoken_without_sw)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 3000, height = 3000, 
                background_color ='black', 
                min_font_size = 10, random_state=100).generate(wc_a) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off")
plt.xlabel('Word Cloud')
plt.tight_layout(pad = 0) 

print("Word Cloud for Roosevelt Speech (after cleaning)!!")
plt.show()








stop_words=nltk.corpus.stopwords.words('english') +list(string.punctuation)

##Dynamic file name with datetime stamp
snapshotdate = datetime.datetime.today().strftime('%d-%m-%Y_%H_%M_%S')
print(snapshotdate)
myfile_name='thresh_xgb_'+snapshotdate+'.csv'
print(myfile_name)
p='E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Competions\\practice hackathon\\submission files\\'
result_xgb.to_csv(p+myfile_name,index=False)

#  ##Save the trained model on disk 
# pickle.dump(xgb_mod2,open("xgbmod_1Jun_99.2Full_Smotte.dat","wb"))
# print("saved model to :xgbmod_1Jun_99.2Full_Smotte.dat")

# # loaded_xgb_mod_disk=pickle.load(open("xgbmod1_withpickle.dat","rb"))
# # print("Loaded from disk : loaded_xgb_mod_disk-from file xgbmod1_withpickle.dat ")

# ##saving on colab 
# path=''
# pickle_out=open(path+"mod.dat","wb")
# pickle.dump(mod,pickle_out)
# pickle_out.close()
