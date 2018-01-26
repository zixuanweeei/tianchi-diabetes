# coding: utf-8

import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.append('../')
from util.feature import add_feature, fillna

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = (train.loc[:, '血糖'] > 7).astype(int)
predictor = XALL.columns
print('Feature: ', XALL.columns.tolist())

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'num_threads': 20,
    'lambda_l1': 0.01,
    'lambda_l2': 0.01,
    'metric': 'auc',
    'verbose': 1,
    'feature_fraction': .9,
    'feature_fraction_seed': 2,
    'bagging_fraction': 0.7,
    'bagging_freq': 0,
    'bagging_seed': 3,
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 1,
    'max_bin': 800,
    'is_unbalance': True
}

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros(XALL.shape[0])
feature_importance = []
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    train_dat = lgb.Dataset(XALL.iloc[train_idx], yALL.iloc[train_idx])
    valid_dat = lgb.Dataset(XALL.iloc[valid_idx], yALL.iloc[valid_idx])

    gbm = lgb.train(params,
                   train_dat,
                   num_boost_round=3000,
                   valid_sets=valid_dat,
                   verbose_eval=100,
                   early_stopping_rounds=100)
    
    preds[valid_idx] = gbm.predict(XALL.iloc[valid_idx],
                                  num_iteration=gbm.best_iteration)
    feature_importance.append(pd.DataFrame(gbm.feature_importance(),
                                          index=predictor, columns=['CV{0}'.format(cv_idx)]))

preds_ = (preds > 0.5).astype(int)
print('Offline auc: {0}'.format(roc_auc_score(yALL, preds_)))
print('Offline accuracy: {0}'.format(accuracy_score(yALL, preds_)))
print('Offline f1: {0}'.format(f1_score(yALL, preds_)))
feature_importance = pd.concat(feature_importance, axis=1)
feature_importance.sort_values(by='CV0', ascending=True, inplace=True)
feature_importance.to_csv('../feature_importance/feature_importance_ctree.csv')
# fig_imp, ax_imp = plt.subplots(figsize=(6, 9*XALL.shape[1]//40))
# feature_importance.plot.barh(ax=ax_imp)
# fig_imp.tight_layout()
# fig_imp.savefig('../feature_importance/feature_importance_ctree.png', dpi=200)
# fig_imp.show()