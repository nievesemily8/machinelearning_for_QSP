#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:07:11 2021
Machine Learning for Sensitivity Analysis of QSP Models 
@author: emilynieves
"""

#  Classification for QSP Virtual patients 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the albumin output QSP model data, baseline variables, and QSP input parameters
output= pd.read_csv('pctchangeFromBaseline_wide.csv') 
albumin_change= output.iloc[:,6]

X = pd.read_csv('thetasim_wide.csv')
X= X.drop('nom_Kf', axis=1)
X= X.drop('pressure_natriuresis_PT_scale', axis=1)
X= X.drop('Na_intake_rate', axis=1)
baseline= pd.read_csv('baselinevals_wide.csv')

combined= pd.concat([baseline,X,albumin_change], axis=1)
#drop cases with unreasonable GFRs
combined= combined.drop(combined[combined.GFR_ml_min>200].index)


#classify albumin change as response or no response 
y= []
for i in range(len(combined.albumin_excretion_rate)):
    if combined.iloc[i,24] <-.3 :
        y += [1]
    else:
        y +=[0]

y=pd.DataFrame(y) 

X= combined.iloc[:,[12,13,14,15,16,17,18,19,20,21,22,23]]

data_set=combined

# Randomly splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Fitting classifiers to the Training set
#uncomment to select from random forest, LR, and SVM
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini')
#classifier= SVC(kernel='linear', probability=True)
#classifier=LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score,log_loss
cm = confusion_matrix(y_test, y_pred)

#feature importances, use importances w/ random forest and coefficients for LR & SVM
importances=classifier.feature_importances_
#coefficients= classifier.coef_


#Get average of feature importances over 1000 different test/train splits
n_iterations=1000
stats= list()
imps= np.empty((12,1000))
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test)
    y_prob=classifier.predict_proba(X_test)
    #fpr,tpr,threshold=roc_curve(y_test,y_prob[:,1],)
    #roc_auc=auc(fpr,tpr)
    #score= auc(fpr, tpr)
    importance= classifier.feature_importances_
    imps[:,i]= importance
    score= log_loss(y_test, y_prob)
    stats.append(score)
    
#find averages of feature importances 
import statistics
importance_averages= list()
for i in range(0,11):
    importance_averages.append(statistics.mean(imps[i]))

#calculate confidence intervals of performance metrics 
n_iterations=1000
stats= list()
imps= np.empty((12,1000))
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test)
    y_prob=classifier.predict_proba(X_test)
    #fpr,tpr,threshold=roc_curve(y_test,y_prob[:,1],)
    #roc_auc=auc(fpr,tpr)
    #score= auc(fpr, tpr)
    score= log_loss(y_test, y_prob)
    stats.append(score)
    
alpha=.95
p=((1-alpha)/2)*100
lower= max(0, np.percentile(stats,p))
p=(alpha+((1-alpha)/2))*100
upper= min(1, np.percentile(stats, p))
print(lower*100, upper*100)



