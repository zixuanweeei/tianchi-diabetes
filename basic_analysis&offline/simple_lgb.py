# coding: utf-8
"""Simple lightGBM model

1. Split train data into offline train set and test set
2. Use all train data to train a new lightGBM model
3. Predict the value of test data
"""

import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print(sys.path)
sys.path.append('../')
from util.feature import add_feature

train = pd.read_csv('../data/d_train_20180102.csv')
train = add_feature(train)


XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']

X_train, X_test, y_train, y_test = train_test_split(XALL, yALL,
                                                   test_size=0.3, random_state=2018)
print('Feature: ', X_train.columns)

all_set = lgb.Dataset(XALL, label=yALL)
train_set = lgb.Dataset(X_train, label=y_train)
test_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 61,
    'num_threads': 2,
    'lambda_l1': 0.01,
    'lambda_l2': 0.01,
    'metric': 'mse',
    'verbose': 0,
    'feature_fraction': 1.0,
    'feature_fraction_seed': 2018,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,
    'bagging_seed': 2018,
}

gbm = lgb.train(params, train_set,
                num_boost_round=5000,
                valid_sets=test_set, valid_names='Test',
                categorical_feature=['性别'],
                early_stopping_rounds=100)
