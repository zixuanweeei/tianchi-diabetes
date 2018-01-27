# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:38:15 2018
@author: wufei
"""

# Parameters
FUDGE_FACTOR = 0.985  # Multiply forecasts by this
XGB_WEIGHT = 0.3200
BASELINE_WEIGHT = 0.0100
OLS_WEIGHT = 0.0620
NN_WEIGHT = 0.0800
CAT_WEIGHT=0.4000
XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models
BASELINE_PRED = 5.631925   # Baseline based on mean of training data, per Oleg

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from catboost import CatBoostRegressor
from tqdm import tqdm
from dateutil.parser import parse

###### READ IN RAW DATA

#

print( "\nReading data from disk ...")
data_path = './'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')

print( "\nProcessing data for LightGBM ..." )

def make_feat_lightgbm(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days    

    data.fillna(data.median(axis=0),inplace=True)

    for c, dtype in zip(data.columns, data.dtypes):
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)    

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]    

    return train_feat,test_feat

df_train,df_test= make_feat_lightgbm(train,test)
x_train = df_train.drop(['id', '血糖'], axis=1)
y_train = df_train['血糖'].values
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)
del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)

##### RUN LIGHTGBM
params = {}

params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3
np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 1000)
del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Preparing x_test...")
x_test = df_test.drop(['id','血糖'], axis=1)

print("\nStart LightGBM prediction ...")
p_test = clf.predict(x_test)
del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )

################
################
##  XGBoost   ##
################
################

#### PROCESS DATA FOR XGBOOST
print( "\nProcessing data for XGBoost ...")
data_path = './'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')

def make_feat_xgb(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days   

    data.fillna(-1,axis=1,inplace=True)
    for c, dtype in zip(data.columns, data.dtypes):
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)    

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]   

    return train_feat,test_feat

df_train,df_test= make_feat_xgb(train,test)
y_train = df_train['血糖'].values
x_train = df_train.drop(['id', '血糖'], axis=1)
y_mean = np.mean(y_train)

print(x_train.shape, y_train.shape)
x_test = df_test.drop(['id','血糖'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

##### RUN XGBOOST
print("\nSetting up data for XGBoost ...")

# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)
print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )

##### RUN XGBOOST AGAIN
print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 150
print("num_boost_rounds="+str(num_boost_rounds))
print( "\nTraining XGBoost again ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
print( "\nPredicting with XGBoost again ...")
xgb_pred2 = model.predict(dtest)
print( "\nSecond XGBoost predictions:" )
print( pd.DataFrame(xgb_pred2).head())

##### COMBINE XGBOOST RESULTS
xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2

#xgb_pred = xgb_pred1
print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )
del df_train
del x_train
del x_test
del dtest
del dtrain
del xgb_pred1
del xgb_pred2 
gc.collect()

######################
######################
##  Neural Network  ##
######################
######################

# Neural network copied from this script:
# Read in data for neural network

print( "\n\nProcessing data for Neural Network ...")
data_path = './'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')

def make_feat_nn(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days    

    data.fillna(-1,axis=1,inplace=True)
    for c, dtype in zip(data.columns, data.dtypes):
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)    

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]    

    return train_feat,test_feat

df_train,df_test= make_feat_nn(train,test)
y_train = df_train['血糖'].values
x_train = df_train.drop(['id', '血糖'], axis=1)
print(x_train.shape, y_train.shape)
x_test = df_test.drop(['id','血糖'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)

## Preprocessing
print("\nPreprocessing neural network data...")
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
len_x=int(x_train.shape[1])
print("len_x is:",len_x)
# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))
print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), batch_size = 32, epochs = 70, verbose=2)
print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_ann = nn.predict(x_test)
print( "\nPreparing results for write..." )
nn_pred = y_pred_ann.flatten()
print( "Type of nn_pred is ", type(nn_pred) )
print( "Shape of nn_pred is ", nn_pred.shape )
print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred).head() )
# Cleanup
del train
del x_train
del x_test
del df_train
del df_test
del y_pred_ann
gc.collect()
################
################
## catboost   ##
################
################
data_path = './'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')
def make_feat_cat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days
    
#    data.fillna(data.median(axis=0),inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    
    return train_feat,test_feat
train_df,test_df = make_feat_cat(train,test)
print('Remove missing data fields ...')
missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))
del num_rows, missing_perc_thresh
gc.collect();
print ("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))
print ("Define training features !!")
exclude_other = ['id','血糖','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))
print ("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 10
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
print ("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)
print ("Training time !!")
X_train = train_df[train_features]
y_train = train_df['血糖']
print(X_train.shape, y_train.shape)
X_test = test_df[train_features]
print(X_test.shape)
num_ensembles = 5
y_pred_cat = 0.0
for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.03,
        depth=6, l2_leaf_reg=3, 
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred_cat += model.predict(X_test)
y_pred_cat /= num_ensembles
del train
del test
gc.collect()
################
################
##    OLS     ##
################
################
np.random.seed(17)
random.seed(17)
print( "\n\nProcessing data for OLS ...")
print( "\nProcessing data for XGBoost ...")
data_path = './'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb18030')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb18030')
def make_feat_ols(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])
    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days
 #   data.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原','乙肝e抗体', '乙肝核心抗体'],axis=1)
    data.fillna(-1,axis=1,inplace=True)
    for c, dtype in zip(data.columns, data.dtypes):
        if dtype == np.float64:
            data[c] = data[c].astype(np.float32)
    
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    
    return train_feat,test_feat
df_train,df_test= make_feat_ols(train,test)
y = df_train['血糖'].values
train = df_train.drop(['id', '血糖'], axis=1)
test = df_test.drop(['id','血糖'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
def MSE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([np.square(y[i]-ypred[i])  for i in range(len(y))]) / (2*len(y))
exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] 
col = [c for c in train.columns if c not in exc]
print("\nFitting OLS...")
reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MSE(y, reg.predict(train)))
rain = [];  y = [] #memory
########################
########################
##  Combine and Save  ##
########################
########################
##### COMBINE PREDICTIONS
FUDGE_FACTOR = 0.9863
print( "\nCombining XGBoost, LightGBM, NN, and baseline predicitons ..." )
lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - NN_WEIGHT - OLS_WEIGHT-CAT_WEIGHT 
lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
cat_weight0= CAT_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)
pred0 = 0
pred0 += xgb_weight0*xgb_pred
pred0 += baseline_weight0*BASELINE_PRED
pred0 += lgb_weight0*p_test
pred0 += nn_weight0*nn_pred
pred0 += cat_weight0*y_pred_cat
print( "\nCombined XGB/LGB/NN/CAT/baseline predictions:" )
print( pd.DataFrame(pred0).head() )
print( "\nPredicting with OLS and combining with XGB/LGB/NN/CAT/baseline predicitons: ..." )
  
pred = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(test) + (1-OLS_WEIGHT)*pred0 )
submission = [float(format(x, '.4f')) for x in pred]
print('predict...')
print( "\nCombined XGB/LGB/NN/CAT/baseline/OLS predictions:" )
#print(submission)
print(MSE(xgb_pred,submission))
submission=pd.DataFrame(submission)
##### WRITE THE RESULTS
from datetime import datetime
print( "\nWriting results to disk ..." )
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,header=False)
print( "\nFinished ...")