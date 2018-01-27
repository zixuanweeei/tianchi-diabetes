# coding: utf-8
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVR

sys.path.append('../')
from util.feature import add_feature, fillna
from util.metric import mse
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test = pd.read_csv('../data/d_test_A_20180102.csv')
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
    
    tree_feature_train = gbm.predict(XALL.iloc[train_idx],
                            num_iteration=gbm.best_iteration,
                            pred_leaf=True)
    regr = ElasticNet(**variables.ElasticNetParams)
    regr.fit(tree_feature_train, yALL.iloc[train_idx])  
    
    test_feature = gbm.predict(test[predictor],
                              pred_leaf=True,
                              num_iteration=gbm.best_iteration)
    test_preds[:, cv_idx] = regr.predict(test_feature)
preds = test_preds.mean(axis=1)

# ============================== STEP 2 =====================================================
rus = RandomUnderSampler(random_state=2018, return_indices=True)
XALL, yALL, idx_resampled = rus.fit_sample(train[predictor], (train['血糖']>7.5).astype(int))
yALL = train.iloc[idx_resampled]['血糖']
XALL = pd.DataFrame(XALL, columns=predictor)
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
    
    tree_feature_train = gbm.predict(XALL.iloc[train_idx],
                            num_iteration=gbm.best_iteration,
                            pred_leaf=True)
    regr = SVR(**variables.SVRParams)
    regr.fit(tree_feature_train, yALL.iloc[train_idx])  
    
    test_feature = gbm.predict(test[predictor],
                              pred_leaf=True,
                              num_iteration=gbm.best_iteration)
    test_preds[:, cv_idx] = regr.predict(test_feature)


step2_preds = test_preds.mean(axis=1)
idx_to_modify = preds > 7.5
preds[idx_to_modify] = step2_preds[idx_to_modify]

submission = pd.DataFrame({'preds': preds})
submission.to_csv('../submission/result_lgb_en.csv', index=False, header=False)
