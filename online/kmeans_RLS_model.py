# coding: utf-8
"""Simple Ridge Regression model

1. Split train data into offline train set and test set
2. Use them for cv
"""

import sys
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn import linear_model

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score

sys.path.append('../')
from util.feature import add_feature, fillna
from util import variables

train = pd.read_csv('../data/d_train_20180102.csv')
test = pd.read_csv('../data/d_test_A_20180102.csv')
test['血糖'] = -1

all_data = pd.concat([train, test], ignore_index=True)
all_data = fillna(all_data)
all_data = add_feature(all_data)

feature_col = [column for column in all_data.columns if column not in ['id', '性别', '体检日期', '血糖']]
scaler = MinMaxScaler()
scaler.fit(all_data.loc[:, feature_col])
all_data.loc[:, feature_col] = scaler.transform(all_data[feature_col])

print(feature_col)
# splits into different catagories
print("Feature embedding ...")
X_embedded = TSNE(n_components=2).fit_transform(all_data[feature_col])
print("Splitting into different catagories ...")
catagories = KMeans(n_clusters=4, random_state=2018).fit_predict(X_embedded)
catagories_for_test = catagories[all_data['血糖'] < 0.0]
print(catagories_for_test)

result = []
for catagory in np.unique(catagories_for_test):
    regressor = linear_model.Ridge(**variables.RidgeParams)

    X_train = all_data.loc[(catagories == catagory) & (all_data['血糖'] >= 0.0), feature_col]
    y_train = all_data.loc[(catagories == catagory) & (all_data['血糖'] >= 0.0), '血糖']
    X_test = all_data.loc[(catagories == catagory) & (all_data['血糖'] < 0.0), feature_col]
    
    regressor.fit(X_train, y_train)

    IDTest = all_data.loc[(catagories == catagory) & (all_data['血糖'] < 0.0), ['id']]
    IDTest.reset_index(drop=True, inplace=True)
    glu = regressor.predict(X_test[feature_col])
    glu = pd.DataFrame(glu.round(2), columns=['glu'])
    result.append(pd.concat([IDTest, glu], axis=1, ignore_index=True))

result = pd.concat(result, ignore_index=True)
result.sort_values(0, inplace=True)
result.loc[:, 1].to_csv('../submission/result_{0}.csv'.format(dt.now().strftime('%Y%m%d_%H%M%S')),
                        index=False, header=False)
