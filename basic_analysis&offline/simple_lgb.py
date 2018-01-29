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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append('../')
from util.feature import add_feature, fillna, nn_feature
from util.metric import mean_square_error
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)
nn_f = nn_feature(train)
train = pd.concat([train, nn_f], axis=1)

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
print('Feature: ', XALL.columns)
scaler = StandardScaler()
XALL = scaler.fit_transform(XALL)

all_set = lgb.Dataset(XALL, label=yALL)
gbm = lgb.cv(variables.lgb_params, all_set,
            num_boost_round=variables.num_boost_round,
            early_stopping_rounds=variables.early_stopping_rounds,
            nfold=5, stratified=False,
            verbose_eval=100,
            feval=mean_square_error,
            seed=2018)
