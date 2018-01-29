# coding: utf-8
# Neural network copied from this script:
# Read in data for neural network

import sys

import numpy as np
import pandas as pd

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
train = fillna(train)
# train = add_feature(train)

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
predictor = XALL.columns
print('Feature: ', XALL.columns.tolist())

sc = StandardScaler()
XALL = sc.fit_transform(XALL)
XALL = pd.DataFrame(XALL, columns=predictor)

# Neural Network
print("\nSetting up neural network model...")
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

kf = KFold(n_splits=5, shuffle=True, random_state=2018)
preds = np.zeros(XALL.shape[0])
nn_f = np.zeros((XALL.shape[0], 26))

for cv_idx, (train_idx, valid_idx) in enumerate(kf.split(XALL)):
    print('CV epoch[{0:2d}]:'.format(cv_idx))
    nn.fit(XALL.iloc[train_idx], yALL.iloc[train_idx], batch_size = 32, epochs = 70, verbose=2)
    y_pred_ann = nn.predict(XALL.iloc[valid_idx])
    preds[valid_idx] = y_pred_ann.flatten()
    nn_f[valid_idx, :] = nn_feature.predict(XALL.iloc[valid_idx])

print('Offline mse: {0}'.format(mean_squared_error(yALL, preds)*0.5))
