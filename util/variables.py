# coding: utf-8

# ================= LightGBM parameters ====================
lgb_params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 15,
    # 'max_depth': 5,
    'num_threads': 2,
    'lambda_l1': 0.01,
    'lambda_l2': 0.01,
    'metric': 'mse',
    'verbose': 1,
    'feature_fraction': 1.0,
    'feature_fraction_seed': 2018,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,
    'bagging_seed': 2018,
    'min_data_in_leaf': 30,
    'min_sum_hessian_in_leaf': 1.e-2,
}

num_boost_round = 1000
early_stopping_rounds=100