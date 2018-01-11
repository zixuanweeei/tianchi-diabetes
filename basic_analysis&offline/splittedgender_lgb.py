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
from util.feature import add_feature, fillna
from util.metric import mean_square_error
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)

# splits into male and female
train_m = train.loc[train['性别'] == 0.0, :]
train_f = train.loc[train['性别'] == 1.0, :]
log = tuple()
scaler = StandardScaler()

for sets in [train_m, train_f]:
    XALL = sets.loc[:, [column for column in train.columns if column not in 
                    ['id', '性别', '体检日期', '血糖']]]
    
    XALL.fillna(XALL.median(), inplace=True)
    columns = XALL.columns

    scaler.fit(XALL)
    XALL = scaler.transform(XALL)
    XALL = pd.DataFrame(XALL, columns=columns)

    yALL = sets.loc[:, '血糖']
    train_set = lgb.Dataset(XALL, label=yALL)

    gbm = lgb.cv(variables.lgb_params, train_set,
                num_boost_round=variables.num_boost_round,
                nfold=5,
                stratified=False,
                early_stopping_rounds=variables.early_stopping_rounds,
                verbose_eval=100,
                feval=mean_square_error,
                seed=2018)
