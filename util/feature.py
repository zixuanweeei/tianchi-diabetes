# coding: utf-8

from itertools import combinations
import multiprocessing

import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures


def GFR(scr, age, gender):
    return 175*scr**-1.154*age**-0.203*(1 - 0.268*gender)

def eGFR(serum, age, gender):
    return np.exp(1.911 + 5.249/serum - 2.114/serum**2 - 0.00686*age - 0.205*gender)

def add_feature(data):
    if is_object_dtype(data['性别']):
        data['性别'] = data['性别'].map({'男':0, '女':1})

    columns_to_poly = [column for column in data.columns if column not in ['性别', 'id', '体检日期', '血糖']]
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    poly_feature = poly.fit_transform(data[columns_to_poly])
    poly_columns = ['%s*%s' % (s1, s2) for s1, s2 in combinations(columns_to_poly, 2)]
    poly_feature = poly_feature[:, len(columns_to_poly):]
    poly_feature = pd.DataFrame(poly_feature, columns=poly_columns)

    log_feature = np.log1p(data[columns_to_poly])
    log_feature.columns=['log_%s' % c for c in columns_to_poly]

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

    data['*天门冬氨酸氨基转换酶strong'] = data['*天门冬氨酸氨基转换酶'].map(lambda x: 1 if (x >= 15) and (x < 40) else 0)
    data['*丙氨酸氨基转换酶strongcolor'] = data['*丙氨酸氨基转换酶'] .map(lambda x: 1 if (x >= 0) and (x < 35) else 0)
    data['*丙氨酸氨基转换酶strongsucce'] = data['*丙氨酸氨基转换酶'] .map(lambda x: 1 if (x >= 4) and (x < 26) else 0)
    data['*碱性磷酸酶strong'] = 0
    data.loc[data['性别'] == 0, '*碱性磷酸酶strong'] = data.loc[data['性别'] == 0, '*碱性磷酸酶'].map(lambda x: 1 if (x > 45) and (x < 125) else 0)
    data.loc[data['性别'] == 1, '*碱性磷酸酶strong'] = data.loc[data['性别'] == 1, '*碱性磷酸酶'].map(lambda x: 1 if (x > 50) and (x < 135) else 0)
    data['*r-谷氨酰基转换酶strong'] = data['*r-谷氨酰基转换酶'].map(lambda x: 1 if x < 50 else 0)
    data['*总蛋白strong'] = data['*总蛋白'].map(lambda x: 1 if (x > 60) and (x < 85) else 0)
    data['白蛋白strong'] = data['白蛋白'].map(lambda x: 1 if (x > 35) and (x < 51) else 0)
    data['*球蛋白strong'] = data['*球蛋白'].map(lambda x: 1 if (x > 20) and (x < 30) else 0)
    data['甘油三酯strong'] = data['甘油三酯'].map(lambda x: 1 if (x > 0.45) and (x < 1.69) else 0)
    data['甘油三酯verystrong'] = data['甘油三酯'].map(lambda x: 1 if (x > 1.70) and (x < 2.25) else 0)
    data['甘油三酯dead'] = data['甘油三酯'].map(lambda x: 1 if x >= 2.26 else 0)
    data['总胆固醇strong'] = data['总胆固醇'].map(lambda x: 1 if (x > 2.85) and (x < 5.69) else 0)
    data['高密度脂蛋白胆固醇strong'] = 0
    data.loc[data['性别'] == 0, '高密度脂蛋白胆固醇strong'] = data.loc[data['性别'] == 0, '高密度脂蛋白胆固醇'].map(lambda x: 1 if x > 1.2 else 0)
    data.loc[data['性别'] == 1, '高密度脂蛋白胆固醇strong'] = data.loc[data['性别'] == 1, '高密度脂蛋白胆固醇'].map(lambda x: 1 if x > 1.4 else 0)
    data['低密度脂蛋白胆固醇strong'] = data['低密度脂蛋白胆固醇'].map(lambda x: 1 if x > 4.14 else 0)
    data['尿素strong'] = data['尿素'].map(lambda x: 1 if (x > 1.5) and (x < 4.4) else 0)
    data['肌酐strong'] = 0
    data.loc[data['性别'] == 0, '肌酐strong'] = data.loc[data['性别'] == 0, '肌酐'].map(lambda x: 1 if (x > 44) and (x < 133) else 0)
    data.loc[data['性别'] == 1, '肌酐strong'] = data.loc[data['性别'] == 1, '肌酐'].map(lambda x: 1 if (x > 70) and (x < 105) else 0)
    data['白细胞计数strong'] = 0
    data.loc[data['年龄'] <= 14, '白细胞计数strong'] = data.loc[data['年龄'] <= 14, '白细胞计数'].map(lambda x: 1 if (x > 5.) and (x < 12.) else 0)
    data.loc[data['年龄'] > 14, '白细胞计数strong'] = data.loc[data['年龄'] > 14, '白细胞计数'].map(lambda x: 1 if (x > 3.5) and (x < 9.5) else 0)
    data['红细胞计数strong'] = 0
    data.loc[data['年龄'] < 6, '红细胞计数strong'] = data.loc[data['年龄'] < 6, '红细胞计数strong'].map(lambda x: 1 if (x > 6.) and (x < 7.) else 0)
    data.loc[(data['年龄'] <= 14)&(data['年龄'] >= 6), '红细胞计数strong'] = data.loc[(data['年龄'] <= 14)&(data['年龄'] >= 6), '红细胞计数strong'].map(lambda x: 1 if (x > 4.2) and (x < 5.2) else 0)
    data.loc[(data['年龄'] > 14)&(data['性别'] == 0), '红细胞计数strong'] = data.loc[(data['年龄'] > 14)&(data['性别'] == 0), '红细胞计数strong'].map(lambda x: 1 if (x > 4.5) and (x < 5.5) else 0)
    data.loc[(data['年龄'] > 14)&(data['性别'] == 1), '红细胞计数strong'] = data.loc[(data['年龄'] > 14)&(data['性别'] == 1), '红细胞计数strong'].map(lambda x: 1 if (x > 4.) and (x < 5.) else 0)
    data['血红蛋白strong'] = 0
    data.loc[data['年龄'] < 6, '血红蛋白strong'] = data.loc[data['年龄'] < 6, '血红蛋白strong'].map(lambda x: 1 if (x > 160.) and (x < 220.) else 0)
    data.loc[(data['年龄'] <= 14)&(data['年龄'] >= 6), '血红蛋白strong'] = data.loc[(data['年龄'] <= 14)&(data['年龄'] >= 6), '血红蛋白strong'].map(lambda x: 1 if (x > 110) and (x < 160) else 0)
    data.loc[(data['年龄'] > 14)&(data['性别'] == 0), '血红蛋白strong'] = data.loc[(data['年龄'] > 14)&(data['性别'] == 0), '血红蛋白strong'].map(lambda x: 1 if (x > 130) and (x < 175) else 0)
    data.loc[(data['年龄'] > 14)&(data['性别'] == 1), '血红蛋白strong'] = data.loc[(data['年龄'] > 14)&(data['性别'] == 1), '血红蛋白strong'].map(lambda x: 1 if (x > 115) and (x < 150) else 0)
    data['红细胞压积strong'] = 0
    data.loc[data['性别'] == 0, '红细胞压积strong'] = data.loc[data['性别'] == 0, '红细胞压积'].map(lambda x: 1 if (x > .4) and (x < .5) else 0)
    data.loc[data['性别'] == 1, '红细胞压积strong'] = data.loc[data['性别'] == 1, '红细胞压积'].map(lambda x: 1 if (x > .35) and (x < .45) else 0)
    data['红细胞平均体积strong'] = 0
    data.loc[data['性别'] == 0, '红细胞平均体积strong'] = data.loc[data['性别'] == 0, '红细胞平均体积'].map(lambda x: 1 if (x > 83) and (x < 93) else 0)
    data.loc[data['性别'] == 1, '红细胞平均体积strong'] = data.loc[data['性别'] == 1, '红细胞平均体积'].map(lambda x: 1 if (x > 82) and (x < 92) else 0)
    data['红细胞平均血红蛋白量strong'] = data['红细胞平均血红蛋白量'].map(lambda x: 1 if (x > 26) and (x < 38) else 0)
    data['红细胞体积分布宽度strong'] = data['红细胞体积分布宽度'].map(lambda x: 1 if x < 14.5 else 0)
    data['血小板计数strong'] = data['血小板计数'].map(lambda x: 1 if (x > 100) and (x < 350) else 0)
    data['血小板平均体积strong'] = data['血小板平均体积'].map(lambda x: 1 if (x > 6.) and (x < 11.5) else 0)
    data['血小板体积分布宽度strong'] = data['血小板体积分布宽度'].map(lambda x: 1 if (x > 15) and (x < 20) else 0)
    data['血小板比积strong'] = data['血小板比积'].map(lambda x: 1 if (x > 0.11) and (x < 0.23) else 0)
    data['中性粒细胞%strong'] = data['中性粒细胞%'].map(lambda x: 1 if (x > 55) and (x < 70) else 0)
    data['淋巴细胞%strong'] = data['淋巴细胞%'].map(lambda x: 1 if (x > 20) and (x < 40) else 0)
    data['单核细胞%strong'] = data['单核细胞%'].map(lambda x: 1 if (x > 3) and (x < 8) else 0)
    data['嗜酸细胞%strong'] = data['嗜酸细胞%'].map(lambda x: 1 if x < 8 else 0)
    data['嗜碱细胞%strong'] = data['嗜碱细胞%'].map(lambda x: 1 if x < 1 else 0)

    data = pd.concat([data, poly_feature, log_feature], axis=1)

    # =============================== Clean feature according to its importance rank =====================
    feature_importance = pd.read_csv('../feature_importance/feature_importance_tree.csv', index_col=0)
    feature_importance = feature_importance.mean(axis=1)
    feature_importance.sort_values(axis=0, ascending=False, inplace=True)
    feature_reserved = feature_importance.head(50).index.tolist()
    feature_reserved.append('血糖')

    return data[feature_reserved]

def fillna(data):
    data = data.drop(columns=['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'])
    if is_object_dtype(data['性别']):
        data['性别'] = data['性别'].map({'男':0, '女':1})   
    feature_col = [column for column in data.columns if column not in ['id', '体检日期', '血糖']]
    
    # feature_min = data[feature_col].min()
    # feature_max = data[feature_col].max()
    # scaled_feature = (data[feature_col] - feature_min) / (feature_max - feature_min)

    # data.loc[:, feature_col] = scaled_feature.values
    columns_na = data.columns[data.isna().sum() > 0]
    complete_sample = data.loc[data.isna().sum(axis=1) == 0, :]
    incomplete_sample = data.loc[data.isna().sum(axis=1) > 0, :]

    params = {
        'objective': 'regression',
        'boosting': 'rf',
        'learning_rate': 0.01,
        'num_leaves': 15,
        'num_threads':  multiprocessing.cpu_count() // 2,
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
    # inverse_values = data[feature_col]*(feature_max - feature_min) + feature_min
    # data.loc[:, feature_col] = inverse_values

    return data


if __name__ == "__main__":
    train = pd.read_csv('../data/d_train_20180102.csv')
    # test = pd.read_csv('../data/d_test_A_20180102.csv')
    # test['血糖'] = -1

    # all_data = pd.concat([train, test], ignore_index=True)
    # filled_data = fillna(all_data)
    train.drop(columns=['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], inplace=True)
    train.fillna(train.median(), inplace=True)
    train = add_feature(train)
    train.to_csv('../data/filled.csv', index=False)

    print(train.shape)
