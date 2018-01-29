# coding: utf-8
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append('../')
from util.feature import add_feature, fillna
# from util.metric import mse
from util import variables

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train = pd.read_csv('../data/d_train_20180102.csv')
test_feature = pd.read_csv('../data/d_test_A_20180102.csv')
test_answer = pd.read_csv('../data/d_answer_a_20180128.csv', header=None)
test_answer.columns = ['血糖']
test = pd.concat([test_feature, test_answer], axis=1)
train = pd.concat([train, test], ignore_index=True)
train = fillna(train)
train = add_feature(train)

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
predictor = XALL.columns
print('Feature: ', XALL.columns.tolist())
# scaler = MinMaxScaler()
# XALL = scaler.fit_transform(XALL)
# XALL = pd.DataFrame(XALL, columns=predictor)

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros(XALL.shape[0])
feature_importance = []
for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    eval_set = Pool(XALL.iloc[valid_idx], yALL.iloc[valid_idx])

    model = CatBoostRegressor(**variables.CatParams)
    model.fit(XALL.iloc[train_idx], yALL.iloc[train_idx],
             eval_set=eval_set, verbose=False)
    
    preds[valid_idx] = model.predict(XALL.iloc[valid_idx])
    # feature_importance.append(pd.DataFrame(gbm.feature_importance(),
    #                                       index=predictor, columns=['CV{0}'.format(cv_idx)]))

print('Offline mse: {0}'.format(mean_squared_error(yALL, preds)*0.5))
# feature_importance = pd.concat(feature_importance, axis=1)
# feature_importance.sort_values(by='CV0', ascending=True, inplace=True)
# feature_importance.to_csv('../feature_importance/feature_importance_tree.csv')
# fig_imp, ax_imp = plt.subplots(figsize=(6, 9*XALL.shape[1]//40))
# feature_importance.plot.barh(ax=ax_imp)
# fig_imp.tight_layout()
# fig_imp.savefig('../feature_importance/feature_importance_tree.png', dpi=200)
# fig_imp.show()
