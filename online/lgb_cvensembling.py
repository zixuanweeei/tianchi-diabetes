# coding: utf-8
import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

sys.path.append('../')
from util.feature import add_feature, fillna
from util.metric import mse
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test_feature = pd.read_csv('../data/d_test_A_20180102.csv')
test_answer = pd.read_csv('../data/d_answer_a_20180128.csv', header=None)
test_answer.columns = ['血糖']
test = pd.concat([test_feature, test_answer], axis=1)
train = pd.concat([train, test], ignore_index=True)

test = pd.read_csv('../data/d_test_B_20180128.csv', encoding='GBK')
test['血糖'] = -1

all_data = pd.concat([train, test], ignore_index=True)
all_data = fillna(all_data)
all_data = add_feature(all_data)

train = all_data.loc[all_data['血糖'] >= 0.0, :]
test = all_data.loc[all_data['血糖'] < 0.0, :]

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
predictor = XALL.columns
print('Feature: ', XALL.columns.tolist())

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros(XALL.shape[0])
feature_importance = []
test_preds = np.zeros((test.shape[0], 5))
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(XALL.iloc[train_idx], yALL.iloc[train_idx])
    valid_dat = lgb.Dataset(XALL.iloc[valid_idx], yALL.iloc[valid_idx])

    gbm = lgb.train(variables.lgb_params,
                   train_dat,
                   num_boost_round=variables.num_boost_round,
                   valid_sets=valid_dat,
                   verbose_eval=100,
                   early_stopping_rounds=variables.early_stopping_rounds,
                   feval=mse) 
    
    test_preds[:, cv_idx] = gbm.predict(test[predictor], num_iteration=gbm.best_iteration)

preds = test_preds.mean(axis=1)
submission = pd.DataFrame({'preds': preds})
submission.to_csv('../submission/result_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                index=False, header=False)
