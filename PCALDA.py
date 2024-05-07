#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:10:32 2021

@author: nitinsinghal
"""

#PCA LDA Learning

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('./RevisedHomesiteTrain1.csv')
test_data = pd.read_csv('./RevisedHomesiteTest1.csv')

# Perform EDA - see the data types, content, statistical properties
print(train_data.describe())
print(train_data.info())
print(train_data.head(5))
print(train_data.dtypes)
      
print(test_data.describe())
print(test_data.info())
print(test_data.head(5))
print(test_data.dtypes)

# Check for % of classification categories
print('train 1 pct: %.2f' %((train_data['QuoteConversion_Flag']==1).sum()/train_data['QuoteConversion_Flag'].count() *100))
print('train 0 pct: %.2f' %((train_data['QuoteConversion_Flag']==0).sum()/train_data['QuoteConversion_Flag'].count() *100))

emptycols = train_data.sum()==0
ec = emptycols[emptycols==True].index
print(len(ec))

# Perform data wrangling - remove duplicate values and clean null values
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Setup the traing and test X, y datasets
X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values
X_test = test_data.iloc[:,:-1].values

# Scale the data as some features have larger range compared to the rest
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Create the XGBClassifier object and fit the training data
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

# Make predictions using the input X test features
y_pred = classifier.predict_proba(X_test)
y_pred1 = y_pred[:,1]

# Output predicted y values into csv file, submit in kaggle competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['QuoteNumber']
df_result['QuoteConversion_Flag'] = y_pred1

df_result.to_csv('./XGBPCA_submission.csv')



