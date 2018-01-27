# coding: utf-8
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.append('../')
from util.feature import add_feature, fillna
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test = pd.read_csv('../data/d_test_A_20180102.csv')
test['血糖'] = -1

all_data = pd.concat([train, test], ignore_index=True)
all_data = fillna(all_data)
all_data = add_feature(all_data)
predictor = [column for column in all_data.columns if column not in ['id', '体检日期', '血糖']]

scaler = MinMaxScaler()
all_data[predictor] = scaler.fit_transform(all_data[predictor])
XALL = all_data.loc[all_data['血糖'] > 0, predictor]
yALL = all_data.loc[all_data['血糖'] > 0, '血糖']

log = tuple()
svr = SVR(**variables.SVRParams)

kf = KFold(n_splits=5, shuffle=False, random_state=2018)
preds = np.zeros(train.shape[0])
feature_importance = []
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_x = XALL.iloc[train_idx]
    train_y = yALL.iloc[train_idx]
    valid_x = XALL.iloc[valid_idx]
    valid_y = yALL.iloc[valid_idx]

    svr.fit(train_x, train_y)        
    preds[valid_idx] = svr.predict(valid_x)

print('Offline mse: {0}'.format(mean_squared_error(train['血糖'], preds)*0.5))
