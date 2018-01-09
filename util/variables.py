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
    'min_data_in_leaf': 100,
    'min_sum_hessian_in_leaf': 1,
}

num_boost_round = 3000
early_stopping_rounds=300


# ================= Bayesian Ridge Regression ==============
BayesianRidgeParams = {'n_iter': 1000, 'tol': 1.e-5, 'alpha_1': 1e-06, 'alpha_2': 1e-06,
                      'lambda_1': 1e-06, 'lambda_2': 1e-06, 'compute_score': False,
                      'fit_intercept': True, 'normalize': True, 
                      'copy_X': True, 'verbose': False}
