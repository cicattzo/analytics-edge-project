import timeit
from sklearn.linear_model import LinearRegression
from optuna.samplers import RandomSampler
import optuna
import sklearn
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



def calculate_final_metrics(model_name, 
                            y_trainvalid, 
                            y_trainvalid_pred, 
                            y_test, 
                            y_test_pred, 
                            hyperparams, 
                            validation_mse):
    results = dict()
    
    results['model'] = model_name
    results['hyperparameters'] = hyperparams
    results['mse_train'] = mean_squared_error(y_trainvalid, y_trainvalid_pred)
    results['mse_test'] = mean_squared_error(y_test, y_test_pred)
    results['mse_validation'] = validation_mse
    
    return (results)

def linear_lasso(x_tr, x_val, x_te, x_trval, 
                 y_tr, y_val, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()

    def objective_lasso(trial):
        lasso_alpha = trial.suggest_loguniform('lasso_alpha', 1e-4, 1e-1)
        model = sklearn.linear_model.Lasso(alpha=lasso_alpha)

        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)

        error = mean_squared_error(y_val, y_pred)

        return error
    
    study_lasso = optuna.create_study(sampler = RandomSampler(seed=1), direction='minimize')
    study_lasso.optimize(objective_lasso, n_trials=n_trials)
    
    best_alpha = study_lasso.best_params['lasso_alpha']
    
    final_model = sklearn.linear_model.Lasso(alpha=best_alpha)
    final_model.fit(x_trval, y_trval)
    
    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)
    
    results = calculate_final_metrics('lasso', 
                                      y_trval, 
                                      y_trval_pred, 
                                      y_te, 
                                      y_test_pred, 
                                      [study_lasso.best_params], 
                                      study_lasso.best_value)
    
    stop = timeit.default_timer()
    
    print('Time for preparing Lasso result: ', round((stop - start)/60.0, 2), ' minutes') 
    
    return results

def linear_ridge(x_tr, x_val, x_te, x_trval, 
                 y_tr, y_val, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()
    
    def objective_ridge(trial):
        ridge_alpha = trial.suggest_loguniform('ridge_alpha', 1e-4, 1e-1)
        model = sklearn.linear_model.Ridge(alpha=ridge_alpha)

        model.fit(x_tr, y_tr)
        y_pred = model.predict(x_val)

        error = mean_squared_error(y_val, y_pred)

        return error
    
    study_ridge = optuna.create_study(sampler = RandomSampler(seed=1), direction='minimize')
    study_ridge.optimize(objective_ridge, n_trials = n_trials)
    
    best_alpha = study_ridge.best_params['ridge_alpha']
    
    final_model = sklearn.linear_model.Ridge(alpha=best_alpha)
    final_model.fit(x_trval, y_trval)
    
    y_trval_pred = final_model.predict(x_trval)
    y_test_pred = final_model.predict(x_te)
    
    results = calculate_final_metrics('ridge', 
                                      y_trval, 
                                      y_trval_pred,  
                                      y_te, 
                                      y_test_pred, 
                                      [study_ridge.best_params], 
                                      study_ridge.best_value)
    
    stop = timeit.default_timer()
    
    print('Time for preparing Ridge result: ', round((stop - start)/60.0, 2), ' minutes') 
    
    return results

def lightgbm_model(categorical_features, 
                      x_tr, x_val, x_te, x_trval, 
                      y_tr, y_val, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()
    
    def objective_lightgbm(trial):
        lgb_train = lgb.Dataset(x_tr, y_tr,
                                categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(x_val, y_val,
                               categorical_feature=categorical_features, reference=lgb_train)

        param = {
            'objective': 'regression_l2',
            'metric': 'mse',
            'boosting_type'   : 'gbdt',
#             'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 1e-1),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 1000, 1000),
            'num_boost_round': trial.suggest_int('num_boost_round', 400, 1000),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 0),
            'min_sum_hessian_in_leaf': trial.suggest_int('min_sum_hessian_in_leaf', 0, 0),
#             'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#             'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#             'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate'   :  trial.suggest_uniform('learning_rate', 1e-5, 1e-1),
        }

        gbm = lgb.train(param, lgb_train)

        preds = gbm.predict(x_val, num_iteration=gbm.best_iteration)

        error = mean_squared_error(y_val, y_pred)
        
        return error
    
    study_lightgbm_v2 = optuna.create_study(sampler = RandomSampler(seed=1), direction='minimize')
    study_lightgbm_v2.optimize(objective_lightgbm, n_trials=n_trials)
    
    best_l2 = study_lightgbm_v2.best_params['lambda_l2']
#     best_num_leaves = study_lightgbm.best_params['num_leaves']
    best_num_boost_round = study_lightgbm_v2.best_params['num_boost_round']
#     best_min_child_samples = study_lightgbm.best_params['min_child_samples']
    best_learning_rate = study_lightgbm_v2.best_params['learning_rate']
    best_max_depth = study_lightgbm_v2.best_params['max_depth']
    best_num_leaves = study_lightgbm_v2.best_params['num_leaves']
    best_min_data_in_leaf = study_lightgbm_v2.best_params['min_data_in_leaf']
    best_min_sum_hessian_in_leaf = study_lightgbm_v2.best_params['min_sum_hessian_in_leaf']
    
    param = {
        'objective': 'rmse',
        'metric': 'rmse',
        'boosting_type'   : 'gbdt',
        'lambda_l2': best_l2,
        'num_leaves': best_num_leaves,
        'min_data_in_leaf': best_min_data_in_leaf,
        'min_sum_hessian_in_leaf': best_min_sum_hessian_in_leaf,
        'num_boost_round': best_num_boost_round,
        'max_depth': best_max_depth,
#         'min_child_samples': best_min_child_samples,
        'learning_rate'   :  best_learning_rate,
    }
    
    lgb_trainvalid = lgb.Dataset(x_trval, y_trval,
                            categorical_feature=categorical_features)
    lgb_test = lgb.Dataset(x_te, y_te,
                           categorical_feature=categorical_features, 
                           reference=lgb_trainvalid)
    
    final_lightgbm = lgb.train(param, lgb_trainvalid)
    
    y_trval_pred = final_lightgbm.predict(x_trval, num_iteration=final_lightgbm.best_iteration)
    y_test_pred = final_lightgbm.predict(x_te, num_iteration=final_lightgbm.best_iteration)
    
    results = calculate_final_metrics('lightgbm_2', 
                                      y_trval, 
                                      y_trval_pred, 
                                      y_te, 
                                      y_test_pred, 
                                      [study_lightgbm_v2.best_params], 
                                      study_lightgbm_v2.best_value)
    
    stop = timeit.default_timer()
    
    print('Time for preparing LightGBM result: ', round((stop - start)/60.0, 2), ' minutes') 
    
    return results

def xgboost_model(x_tr, x_val, x_te, x_trval, 
                   y_tr, y_val, y_te, y_trval, n_trials):
    
    start = timeit.default_timer()
    
    def objective_xgboost(trial):
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)

        param = {
            "silent": 1,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1e-4),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1e-3),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "eta": trial.suggest_loguniform("eta", 0.0001, 0.1),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0)
        }

    #     param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    #     if param["booster"] == "dart":
    #         param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
    #         param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
    #         param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
    #         param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
        bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
        preds = bst.predict(dvalid)

        error = mean_squared_error(y_val, y_pred)

        return error
    
    study_xgboost = optuna.create_study(sampler = RandomSampler(seed=1), direction='minimize')
    study_xgboost.optimize(objective_xgboost, n_trials=n_trials)
    
    best_lambda = study_xgboost.best_params['lambda']
    best_alpha = study_xgboost.best_params['alpha']
    best_max_depth = study_xgboost.best_params['max_depth']
    best_eta = study_xgboost.best_params['eta']
    best_gamma = study_xgboost.best_params['gamma']
    
    param = {
        "silent": 1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": best_lambda,
        "alpha": best_alpha,
        "max_depth": best_max_depth,
        "eta": best_eta,
        "gamma": best_gamma
    }
    
    d_trainvalid = xgb.DMatrix(x_trval, label=y_trval)
    d_test = xgb.DMatrix(x_te, label=y_te)
    
    final_xgboost = xgb.train(param, d_trainvalid)
    
    y_trval_pred = final_xgboost.predict(d_trainvalid)
    y_test_pred = final_xgboost.predict(d_test)
    
    results = calculate_final_metrics('xgboost', 
                                      y_trval, 
                                      y_trval_pred, 
                                      y_te, 
                                      y_test_pred, 
                                      [study_xgboost.best_params], 
                                      study_xgboost.best_value)
    
    stop = timeit.default_timer()
    
    print('Time for preparing XGBoost result: ', round((stop - start)/60.0, 2), ' minutes') 
    
    return results



