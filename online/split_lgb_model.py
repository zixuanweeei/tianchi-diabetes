# coding: utf-8
"""Simple lightGBM model

1. Split train data into offline train set and test set
2. Use all train data to train a new lightGBM model
3. Predict the value of test data
"""

import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append('../')
from util.feature import add_feature
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test = pd.read_csv('../data/d_test_A_20180102.csv')
test['血糖'] = -1

all_data = pd.concat([train, test], ignore_index=True)
all_data = fillna(all_data)
all_data = add_feature(all_data)

feature_col = [column for column in all_data.columns if column not in ['id', '性别', '体检日期', '血糖']]
scaler = MinMaxScaler()
scaler.fit(all_data.loc[:, feature_col])
all_data.loc[:, feature_col] = scaler.transform(all_data[feature_col])

train = all_data.loc[all_data['血糖'] >= 0.0, :]
test = all_data.loc[all_data['血糖'] < 0.0, :]

# splits into male and female
train_m = train.loc[train['性别'] == 0, :]
train_f = train.loc[train['性别'] == 1, :]
test_m = test.loc[test['性别'] == 0, :]
test_f = test.loc[test['性别'] == 1, :]

result = []

for train_sets, test_sets in [(train_m, test_m), (train_f, test_f)]:
    XALL = train_sets.loc[:, feature_col]

    yALL = train_sets.loc[:, '血糖']
    train_set = lgb.Dataset(XALL, label=yALL)

    gbm = lgb.train(variables.lgb_params, train_set,
                    num_boost_round=variables.num_boost_round,
                    valid_sets=train_set, valid_names='Self',
                    early_stopping_rounds=100)

    IDTest = test_sets.loc[:, ['id']]
    IDTest.reset_index(drop=True, inplace=True)
    glu = gbm.predict(test_sets[feature_col], num_iteration=gbm.best_iteration)
    glu = pd.DataFrame(glu.round(2), columns=['glu'])
    result.append(pd.concat([IDTest, glu], axis=1, ignore_index=True))

result = pd.concat(result, ignore_index=True)
result.sort_values(0, inplace=True)
result.loc[:, 1].to_csv('../submission/result_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                        index=False, header=False)
