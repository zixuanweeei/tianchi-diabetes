from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import pandas as pd
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

n_folds=5;

def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])

    data['性别'] = data['性别'].map({'男': 1, '女': 0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days

    data=data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat, test_feat


def getscale(data):
    bound = data['血糖'].median() * 2 - data['血糖'].min()
    a = data[data['血糖'] > bound]
    t = a['血糖'].mean()/bound
    return t;

def rmsle_cv(model):
    kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
    rmse= np.sqrt(-cross_val_score(model, train_feat.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

data_path = './data/'

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train = pd.read_csv(data_path + 'd_train_20180102.csv', encoding='utf-8');
test = pd.read_csv(data_path + 'd_test_A_20180102.csv', encoding='utf-8');

# data_pre = pd.DataFrame()
# prelable=pd.read_csv(data_path+'preclassification.csv',encoding='utf-8');

y_train=train['血糖'];
train_feat, test_feat = make_feat(train, test)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1));
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3));
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5);
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5);

predictors = [f for f in test_feat.columns if f not in ['血糖']]

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
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'], categorical_feature=['性别'])
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
    test_preds[:, i] = gbm.predict(test_feat[predictors])

a=test_preds.mean(axis=1);
submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso));

averaged_models.fit(train_feat.values,y_train);
stacked_train_pred=averaged_models.predict(train_feat.values);
stacked_pred = averaged_models.predict(test_feat.values);



c =a*0.9+stacked_pred*0.1;
t=getscale(train_feat);
submission = pd.DataFrame({'pred': c})
bound=submission.mean()*2.1-submission.min()
submission[submission[['pred']]>bound]=submission[submission[['pred']]>bound]*t;
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds*0.9+stacked_train_pred*0.1) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))
