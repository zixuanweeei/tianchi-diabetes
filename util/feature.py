# coding: utf-8

import pandas as pd
import numpy as np

def GFR(scr, age, gender):
    return 175*scr**-1.154*age**-0.203*(1 - 0.268*gender)

def eGFR(serum, age, gender):
    return np.exp(1.911 + 5.249/serum - 2.114/serum**2 - 0.00686*age - 0.205*gender)

def add_feature(data):
    data['性别'] = data['性别'].map({'男':0, '女':1})
    data['体检日期'] = pd.to_datetime(data['体检日期'], format='%d/%m/%Y')
    data['weekday'] = data['体检日期'].dt.dayofweek
    data['month'] = data['体检日期'].dt.month
    data['dayofyear'] = data['体检日期'].dt.dayofyear
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
    data.drop(columns=['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], inplace=True)

    return data
