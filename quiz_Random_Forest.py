# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:43:23 2021

@author: Abhinav.Bajpai
"""

##QUIZ_Random forest
azca_orig=pd.read_csv('E:\\AbhinavB\\Kaggle\\Python related\\DSBA related\\Week 12-RandomForest\\azcabgptca.csv')

azca=azca_orig.copy()

azca.drop(azca.columns[azca.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

azca['died'].value_counts()
##Imbalanced data set 
     
 ##Duplicates 
dups=azca.duplicated()     
print("No of duplicates in this dataset=",dups.sum())
azca[dups]     
azca.drop_duplicates(inplace=True)       
##Outliers
azca.boxplot()
plt.xticks(rotation=90);
##outliers exist but CART and RF not sensitive to outliers

##Missing values 
azca.isnull().sum()
##Scaling -->doesnot impact CART and RF
##Multicollinearity 
plt.figure(figsize=(12,7))
sns.heatmap(azca.corr(),annot=True,fmt='.2f',cmap='rainbow')
plt.show()
##data prep for RF
all_ind_ds=azca.drop(["died"],axis=1)
all_labels=azca.pop("died")

# splitting data into training and test set for independent attributes
trainset,testset, train_labels, test_labels = train_test_split(all_ind_ds, all_labels, test_size=.30, random_state=0)
print(testset.shape)
print(len(test_labels))
print(trainset.shape)
print(len(train_labels))

##Baseline model
all_labels.value_counts()
##Baseline dumb model which predicts all NO/0 for target  will have acc=96% 
##Our model should at least beat pure guesswork model

########RANDOM FOREST (Default parameters)#####################
##Creating rf model with default paramters (IMPUTED with Median)
rfc_obj3=RandomForestClassifier(random_state=0,n_estimators=500)
rfc_mod3=rfc_obj3.fit(trainset,train_labels)
pred_test_normal=rfc_mod3.predict(testset)
confusion_matrix(test_labels,pred_test_normal)
acc_normal=(575+5)/(575+5+6+2)
print(acc_normal)
rfc_mod3.score(testset,test_labels)

pred_train_normal=rfc_mod3.predict(trainset)
confusion_matrix(train_labels,pred_train_normal)
rfc_mod3.score(trainset,train_labels)
print(classification_report(test_labels,pred_test_normal))
# AUC and ROC for the test data
# predict probabilities
probs3 = rfc_mod3.predict_proba(testset)
# keep probabilities for the positive outcome only
probs3 = probs3[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc3 = roc_auc_score(test_labels, probs3)
print('AUC for default parameters model on test set: %.3f' % auc3)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr3, tpr3, thresholds3 = roc_curve(test_labels, probs3)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr3, tpr3, marker='.')
# show the plot
plt.show()

##RF with grid search ####
param_grid = {
    'max_depth': [7, 10],
    'max_features': [0,4,6,0.2],
    'min_samples_leaf': [50, 100],
    'min_samples_split': [150, 300],
    'n_estimators': [300, 500]
}
rfc_obj4=RandomForestClassifier(random_state=0)
rfc_mod_gs4=GridSearchCV(estimator=rfc_obj4,param_grid=param_grid,cv=3)
rfc_mod_gs4.fit(trainset,train_labels)
rfc_mod_gs4.best_params_
rfc_mod_bst4=rfc_mod_gs4.best_estimator_
pred_test_gs4=rfc_mod_bst4.predict(testset)
confusion_matrix(test_labels,pred_test_gs4)
acc_gs=(1013+0)/(1013+0+204+0)
print(acc_gs)
print(classification_report(test_labels,pred_test_normal))
# AUC and ROC for the test data
# predict probabilities
probs2 = rfc_mod_bst.predict_proba(testset)
# keep probabilities for the positive outcome only
probs2 = probs2[:, 1]
# calculate AUC
from sklearn.metrics import roc_auc_score
auc2 = roc_auc_score(test_labels, probs2)
print('AUC for grid searched model: %.3f' % auc2)
# calculate roc curve
from sklearn.metrics import roc_curve
fpr2, tpr2, thresholds2 = roc_curve(test_labels, probs2)
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr2, tpr2, marker='.')
# show the plot
plt.show()