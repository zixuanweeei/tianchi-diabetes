# coding: utf-8
# coding: utf-8
"""Simple Bayesian Ridge Regression model
"""

import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

sys.path.append('../')
from util.feature import add_feature, fillna
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test = pd.read_csv('../data/d_test_A_20180102.csv')
test['血糖'] = -1

all_data = pd.concat([train, test], ignore_index=True)
all_data = fillna(all_data)
all_data = add_feature(all_data)

feature_columns = [column for column in all_data.columns if column not in ['id', '性别', '体检日期', '血糖']]
scaler = MinMaxScaler()
scaler.fit(all_data.loc[:, feature_columns])
all_data.loc[:, feature_columns] = scaler.transform(all_data[feature_columns])

train = all_data.loc[all_data['血糖'] >= 0.0, :]
test = all_data.loc[all_data['血糖'] < 0.0, :]

# splits into male and female
train_m = train.loc[train['性别'] == 0, :]
train_f = train.loc[train['性别'] == 1, :]
test_m = test.loc[test['性别'] == 0, :]
test_f = test.loc[test['性别'] == 1, :]

regressor = linear_model.BayesianRidge(**variables.BayesianRidgeParams)
result = []

for train_sets, test_sets in [(train_m, test_m), (train_f, test_f)]:
    XALL = train_sets.loc[:, feature_columns]
    yALL = train_sets.loc[:, '血糖']

    regressor.fit(XALL, yALL)

    IDTest = test_sets.loc[:, ['id']]
    IDTest.reset_index(drop=True, inplace=True)
    glu = regressor.predict(test_sets[feature_columns])
    glu = pd.DataFrame(glu.round(2), columns=['glu'])
    result.append(pd.concat([IDTest, glu], axis=1, ignore_index=True))

result = pd.concat(result, ignore_index=True)
result.sort_values(0, inplace=True)
result.loc[:, 1].to_csv('../submission/result_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                        index=False, header=False)