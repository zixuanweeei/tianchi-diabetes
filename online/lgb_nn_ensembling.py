# coding: utf-8

import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
import lightgbm as lgb

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
predictor = [column for column in all_data.columns if column not in ['id', '体检日期', '血糖']]
print('Neural network\'s feature: ', all_data.columns.tolist())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data[predictor])

XALL = scaled_data[all_data['血糖'] >= 0.0, :]
yALL = all_data.loc[all_data['血糖'] >= 0.0, '血糖'].values
test = scaled_data[all_data['血糖'] < 0.0, :]

# ============================== NN model ==========================================
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer='normal', input_dim=XALL.shape[1]))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer='normal'))
nn.add(PReLU())
nn.add(BatchNormalization(name='nn_feature'))
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))

nn_feature = Model(inputs=nn.input,
                  outputs=nn.get_layer('nn_feature').output)

nn.fit(XALL, yALL, batch_size = 32, epochs = 70, verbose=2)
y_pred_ann = nn.predict(test)
preds_nn = y_pred_ann.flatten()

# ========================= LightGBM Model ======================================
nn_feat = nn_feature.predict(scaled_data)
nn_feat = pd.DataFrame(nn_feat, columns=['nn_%d' % column for column in range(26)])
all_data = add_feature(all_data)
all_data = pd.concat([all_data, nn_feat], axis=1)

train = all_data.loc[all_data['血糖'] >= 0.0, :]
test = all_data.loc[all_data['血糖'] < 0.0, :]

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
predictor = XALL.columns
print('LightGBM\'s feature: ', XALL.columns.tolist())

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
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

preds_lgb = test_preds.mean(axis=1)

preds = (preds_nn + preds_lgb)/2
submission = pd.DataFrame({'preds': preds})
submission.to_csv('../submission/result_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                 index=False, header=False)
