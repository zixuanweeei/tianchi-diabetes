# coding: utf-8

# ================= LightGBM parameters ====================
lgb_params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 128,
    # 'max_depth': 5,
    'num_threads': 2,
    'lambda_l1': 0.01,
    'lambda_l2': 0.01,
    'metric': 'mse',
    'verbose': 1,
    'feature_fraction': .8,
    'feature_fraction_seed': 2,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'bagging_seed': 3,
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 1.e-2,
}

num_boost_round = 10000
early_stopping_rounds=300


# ================= Bayesian Ridge Regression ==============
BayesianRidgeParams = {'n_iter': 1000, 'tol': 1.e-5, 'alpha_1': 1e-06, 'alpha_2': 1e-06,
                      'lambda_1': 1e-06, 'lambda_2': 1e-06, 'compute_score': False,
                      'fit_intercept': True, 'normalize': True, 
                      'copy_X': True, 'verbose': False}

# ==================== Ridge Regeression ===================
RidgeParams = {'alpha': 1.0, 'fit_intercept': True, 
              'normalize': False, 'copy_X': True, 
              'max_iter': None, 'tol': 0.001, 
              'solver': 'auto', 'random_state': None}
