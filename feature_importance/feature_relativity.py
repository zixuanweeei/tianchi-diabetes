# coding: utf-8
import sys

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.append('../')
from util.feature import fillna, add_feature

train = pd.read_csv('../data/d_train_20180102.csv')
train = fillna(train)
train = add_feature(train)

XALL = train.loc[:, [column for column in train.columns if column not in ['id', '体检日期', '血糖']]]
yALL = train.loc[:, '血糖']
predictor = XALL.columns

kendall_coef = []
for col in XALL:
    kendall_coef.append(kendalltau(XALL[col], yALL))
kendall_coef = pd.DataFrame(kendall_coef, columns=['tau', 'p'], index=XALL.columns)

pearsonr_coef = []
for col in XALL:
    pearsonr_coef.append(pearsonr(XALL[col], yALL))
pearsonr_coef = pd.DataFrame(pearsonr_coef, columns=['pearson', 'p'], index=XALL.columns)

spearmanr_coef = []
for col in XALL:
    spearmanr_coef.append(spearmanr(XALL[col], yALL))
spearmanr_coef = pd.DataFrame(spearmanr_coef, columns=['spearmanr', 'p'], index=XALL.columns)
