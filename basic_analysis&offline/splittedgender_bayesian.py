# coding: utf-8
"""Simple Bayesian Ridge Regression model

1. Split train data into offline train set and test set
2. Use them for cv
"""

import sys

import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score

sys.path.append('../')
from util.feature import add_feature, fillna
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)

# splits into male and female
train_m = train.loc[train['性别'] == 0, :]
train_f = train.loc[train['性别'] == 1, :]
log = tuple()
# scaler = MinMaxScaler()
scaler = StandardScaler()
regressor = linear_model.BayesianRidge(**variables.BayesianRidgeParams)

for sets in [train_m, train_f]:
    XALL = sets.loc[:, [column for column in train.columns if column not in 
                    ['id', '性别', '体检日期', '血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']]]
    
    XALL.fillna(XALL.median(), inplace=True)
    columns = XALL.columns

    scaler.fit(XALL)
    XALL = scaler.transform(XALL)
    XALL = pd.DataFrame(XALL, columns=columns)

    yALL = sets.loc[:, '血糖']

    scores = cross_val_score(regressor, XALL, yALL, cv=5, scoring='neg_mean_squared_error')
    print("Accuracy: %0.2f (+/- %0.2f)" % ((scores/2).mean(), (scores/2).std() * 2))
