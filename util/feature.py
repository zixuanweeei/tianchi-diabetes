# coding: utf-8

import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np

import lightgbm as lgb


def GFR(scr, age, gender):
    return 175*scr**-1.154*age**-0.203*(1 - 0.268*gender)

def eGFR(serum, age, gender):
    return np.exp(1.911 + 5.249/serum - 2.114/serum**2 - 0.00686*age - 0.205*gender)

def add_feature(data):
    if is_object_dtype(data['性别']):
        data['性别'] = data['性别'].map({'男':0, '女':1})
    data['体检日期'] = pd.to_datetime(data['体检日期'], format='%d/%m/%Y')
    data['weekday'] = data['体检日期'].dt.dayofweek
    # data['month'] = data['体检日期'].dt.month
    # data['dayofyear'] = data['体检日期'].dt.dayofyear
    data['白蛋白/总蛋白'] = data['白蛋白']/data['*总蛋白']
    data['球蛋白/总蛋白'] = data['*球蛋白']/data['*总蛋白']
    data['甘油三酯/总胆固醇'] = data['甘油三酯']/data['总胆固醇']
    data['高低固醇比例'] = data['高密度脂蛋白胆固醇']/data['低密度脂蛋白胆固醇']
    data['尿素酸比例'] = data['尿素']/data['尿酸']
    data['白红细胞比例'] = data['白细胞计数']/data['红细胞计数']
    data.loc[data['嗜酸细胞%'] == 0, ['嗜酸细胞%']] = 0.01
    data['嗜碱酸细胞比例'] = data['嗜碱细胞%']/data['嗜酸细胞%']
    data['年龄段'] = data['年龄'] // 5
    # data['表面抗原/表面抗体'] = data['乙肝表面抗原']/data['乙肝表面抗体']
    # data['e抗原/e抗体'] = data['乙肝e抗原']/data['乙肝e抗体']
    # data['表面抗原/核心抗体'] = data['乙肝表面抗原']/data['乙肝核心抗体']
    # data['e抗原/核心抗体'] = data['乙肝e抗原']/data['乙肝核心抗体']
    data['eGFR'] = eGFR(data['肌酐'], data['年龄'], data['性别'])
    data['GFR'] = GFR(data['肌酐'], data['年龄'], data['性别'])

    return data

def fillna(data):
    data = data.drop(columns=['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'])
    if is_object_dtype(data['性别']):
        data['性别'] = data['性别'].map({'男':0, '女':1})   
    feature_col = [column for column in data.columns if column not in ['id', '体检日期', '血糖']]
    
    feature_min = data[feature_col].min()
    feature_max = data[feature_col].max()
    scaled_feature = (data[feature_col] - feature_min) / (feature_max - feature_min)

    data.loc[:, feature_col] = scaled_feature.values
    columns_na = data.columns[data.isna().sum() > 0]
    complete_sample = data.loc[data.isna().sum(axis=1) == 0, :]
    incomplete_sample = data.loc[data.isna().sum(axis=1) > 0, :]

    params = {
        'objective': 'regression',
        'boosting': 'rf',
        'learning_rate': 0.01,
        'num_leaves': 15,
        'num_threads': 2,
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 1e-2,
        'feature_fraction': 0.7,
        'feature_fraction_seed': 2018,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'bagging_seed': 2018,
        'tree_learner': 'feature',
        'verbose': -1,
        'metric': 'mse',
    }

    for target in columns_na:
        X = complete_sample.loc[:, [column for column in feature_col if column is not target]]
        y = complete_sample.loc[:, target]
        train_set = lgb.Dataset(X, label=y)
        
        gbm = lgb.train(params, train_set,
                       num_boost_round=1000,
                       categorical_feature=['性别'],
                       valid_sets=train_set, valid_names='train',
                       early_stopping_rounds=300,
                       verbose_eval=False)
        XTest = incomplete_sample.loc[incomplete_sample[target].isna(), feature_col].values
        na_sample_idxer = incomplete_sample[target].isna()
        result_to_fill = gbm.predict(XTest, num_iteration=gbm.best_iteration)
        incomplete_sample.loc[na_sample_idxer, target] = result_to_fill
    
    data = pd.concat([complete_sample, incomplete_sample])
    inverse_values = data[feature_col]*(feature_max - feature_min) + feature_min
    data.loc[:, feature_col] = inverse_values

    return data


if __name__ == "__main__":
    train = pd.read_csv('../data/d_train_20180102.csv')
    test = pd.read_csv('../data/d_test_A_20180102.csv')
    test['血糖'] = -1

    all_data = pd.concat([train, test], ignore_index=True)
    filled_data = fillna(all_data)
    filled_data.to_csv('../data/filled.csv', index=False)

    print(filled_data.columns[filled_data.isna().sum() > 0])
