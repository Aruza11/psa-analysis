import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier

from together_utils.fairness_functions import compute_fairness
from together_utils.model_selection import nested_cross_validate


def XGB(X,Y,
        learning_rate=None, depth=None, estimators=None, gamma=None, child_weight=None, subsample=None,
        seed=None):

    ### model & parameters
    indicator = "xgb"
    c_grid = {"learning_rate": learning_rate,
              "max_depth": depth,
              "n_estimators": estimators,
              "gamma": gamma,
              "min_child_weight": child_weight,
              "subsample": subsample}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  indicator = indicator,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'combined_holdout_auc': combined_holdout_auc,
            'ky_holdout_auc': ky_holdout_auc,
            'fl_holdout_auc': fl_holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def RF(X, Y,
       depth=None, estimators=None, impurity=None,
       seed=None):

    ### model & parameters
    indicator = "rf"
    c_grid = {"n_estimators": estimators,
              "max_depth": depth,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  indicator = indicator,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'combined_holdout_auc': combined_holdout_auc,
            'ky_holdout_auc': ky_holdout_auc,
            'fl_holdout_auc': fl_holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}




def LinearSVM(X, Y,
              C, 
              seed=None):
    
    ### model & parameters
    indicator = "svm"
    index = 'svm'
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
    combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  indicator = indicator,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed,
                                                                                  index = index)
    return {'best_param': best_param,
            'combined_holdout_auc': combined_holdout_auc,
            'ky_holdout_auc': ky_holdout_auc,
            'fl_holdout_auc': fl_holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}

def Lasso(X, Y,
          C,
          seed=None):
    
    ### model & parameters
    indicator = "lasso"
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  indicator = indicator,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'combined_holdout_auc': combined_holdout_auc,
            'ky_holdout_auc': ky_holdout_auc,
            'fl_holdout_auc': fl_holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def Logistic(X, Y,
             C,
             seed=None):
    
    ### model & parameters
    indicator = "logistic"
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  indicator=indicator,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'combined_holdout_auc': combined_holdout_auc,
            'ky_holdout_auc': ky_holdout_auc,
            'fl_holdout_auc': fl_holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}