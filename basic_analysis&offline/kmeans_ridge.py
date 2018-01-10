# coding: utf-8
"""Simple Ridge Regression model

1. Split train data into offline train set and test set
2. Use them for cv
"""

import sys

import numpy as np
import pandas as pd
from sklearn import linear_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score

sys.path.append('../')
from util.feature import add_feature, fillna
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)
feature_col = [column for column in train.columns if column not in ['id', '体检日期', '血糖']]
print(feature_col)
# splits into different catagories
print("Feature embedding ...")
X_embedded = TSNE(n_components=2).fit_transform(train[feature_col])
print("Splitting into different catagories ...")
catagories = KMeans(n_clusters=4, random_state=2018).fit_predict(X_embedded)

# scaler = MinMaxScaler()
scaler = StandardScaler()

print("Start CV-ing ...")
for catagory in np.unique(catagories):
    XALL = train.loc[catagories == catagory, feature_col]    
    XALL.fillna(XALL.median(), inplace=True)
    columns = XALL.columns

    scaler.fit(XALL)
    XALL = scaler.transform(XALL)
    XALL = pd.DataFrame(XALL, columns=columns)
    yALL = train.loc[catagories == catagory, '血糖']

    regressor = linear_model.Ridge(**variables.RidgeParams)
    scores = cross_val_score(regressor, XALL, yALL, cv=5, scoring='neg_mean_squared_error')
    print("Accuracy: %0.2f (+/- %0.2f)" % ((scores/2).mean(), (scores/2).std() * 2))