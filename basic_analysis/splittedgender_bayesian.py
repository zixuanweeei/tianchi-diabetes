# coding: utf-8
"""Simple lightGBM model

1. Split train data into offline train set and test set
2. Use all train data to train a new lasso model
3. Predict the value of test data
"""

import sys

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

sys.path.append('../')
from util.feature import add_feature

train = pd.read_csv('../data/d_train_20180102.csv')
train = add_feature(train)

# splits into male and female
train_m = train.loc[train['性别'] == 0, :]
train_f = train.loc[train['性别'] == 1, :]
log = tuple()
scaler = MinMaxScaler()
regressor = linear_model.BayesianRidge()

for sets in [train_m, train_f]:
    XALL = sets.loc[:, [column for column in train.columns if column not in 
                    ['id', '性别', '体检日期', '血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']]]
    
    XALL.fillna(XALL.median(), inplace=True)
    columns = XALL.columns

    scaler.fit(XALL)
    XALL = scaler.transform(XALL)
    XALL = pd.DataFrame(XALL, columns=columns)

    yALL = sets.loc[:, '血糖']

    X_train, X_test, y_train, y_test = train_test_split(XALL, yALL,
                                                        test_size=0.3, random_state=2018)

    regressor.fit(X_train, y_train)
    pred_train = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)
    log += ((mse(y_train, pred_train), mse(y_test, pred_test)), )
    
for score1, score2 in log:
    print('train\'s mse: {0}, test\'s mse: {1}'.format(score1, score2))
