import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_fairness
from utils.model_selection import nested_cross_validate
from interpret.glassbox import ExplainableBoostingClassifier


def EBM(X,Y, learning_rate=None, depth=None,estimators=None, holdout_split=None, seed=None):
    ### model & parameters
    ebm = ExplainableBoostingClassifier(random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate, 
              "holdout_split": holdout_split}
    
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
    #holdout_auc, best_param, auc_diffs, fairness_overview, confusion_matrix_rets = nested_cross_validate(X=X,
    #                                                                                                     Y=Y,
    #                                                                                                     estimator=ebm,
    #                                                                                                     c_grid=c_grid,
    #                                                                                                     seed=seed)
    
    summary = nested_cross_validate(X=X, Y=Y, estimator=ebm, c_grid=c_grid, seed=seed)
    return summary

def CART(X, Y,
         depth=None, split=None, impurity=None,
         seed=None):

    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    c_grid = {"max_depth": depth,
              "min_samples_split": split,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    #holdout_auc, best_param, auc_diffs, fairness_overview, confusion_matrix_rets = nested_cross_validate(X=X,
    #                                                                                                     Y=Y,
    #                                                                                                     estimator=cart,
    #                                                                                                     c_grid=c_grid,
    #                                                                                                     seed=seed)
    
    summary = nested_cross_validate(X=X, Y=Y, estimator=cart, c_grid=c_grid, seed=seed)   
    return summary