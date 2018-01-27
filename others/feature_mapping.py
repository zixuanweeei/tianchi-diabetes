import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import KFold
from scipy.stats import pearsonr

train = pd.read_csv('./data/d_train_20180102.csv', encoding="utf-8")
test = pd.read_csv('./data/d_test_A_20180102.csv', encoding="utf-8")

# 将特征进行多项式映射
def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])
    # print(data.shape)
    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    data = data.drop(['体检日期'], 1, inplace=False) #删除了体检日期
    #data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days

    data.fillna(data.median(axis=0),inplace=True)

    predictors = [f for f in data.columns if f not in ['血糖']]
    train_target = data[data.id.isin(train_id)]['血糖']

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    poly_data = poly.fit_transform(data[predictors])

    train_feat = poly_data[data.id.isin(train_id)]
    test_feat = poly_data[data.id.isin(test_id)]

    return train_feat, train_target, test_feat

train_feat, train_target, test_feat = make_feat(train,test)


def correlation(train_feat,train_target):
    coef = []
    for i in np.arange(train_feat.shape[1]):
        coef.append(np.array(pearsonr(train_feat[:,i],train_target))[0])
    return coef

corr = correlation(train_feat, train_target)
index = np.arange(len(corr))
corr_list = list(zip(index, corr))
b = np.array(corr_list)
index_desc = np.lexsort(-b.T)  # 挑出降序排列的索引
# print(index_desc[0])

#挑选出相关系数大的索引
def select_feature(feature,n):
    feat_new = np.zeros((feature.shape[0], 1))
    for i in np.arange(n):
        feat_new = np.column_stack((feat_new, feature[:, index_desc[i]]))
    feat_new = np.array([[row[i] for i in np.arange(feat_new.shape[1]) if i != 0] for row in feat_new])
    return feat_new

train_feat_new = select_feature(train_feat,100)
test_feat_new = select_feature(test_feat,100)

#转换成DataFrame格式
train_feat_df = pd.DataFrame(train_feat_new)
train_target_df = pd.DataFrame(train_target)
train_df = pd.concat([train_feat_df,train_target_df],axis=1)
test_df = pd.DataFrame(test_feat_new)
# print(train_df['血糖'])
# print(train_df.iloc[:,-1])
# print(train_feat_df.shape)
# print(test_feat_df.shape)

predictors = [f for f in test_df.columns if f not in ['血糖']]
# print(predictors)
def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_df.shape[0])
test_preds = np.zeros((test_df.shape[0], 5))

kf = KFold(len(train_df), n_folds=5, shuffle=True, random_state=520)

for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_df.iloc[train_index]
    train_feat2 = train_df.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:, i] = gbm.predict(test_df[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_df['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))

submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')

