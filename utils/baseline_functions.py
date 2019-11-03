import numpy as np
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils.model_selection import nested_cross_validate

#### XGBoost ####
def XGB(X,
        Y,
        learning_rate=None, 
        depth=None, 
        estimators=None, 
        gamma=None, 
        child_weight=None, 
        subsample=None,
        seed=None):

    ### model 
    xgboost = xgb.XGBClassifier(random_state=seed)
    
    ## parameters
    c_grid = {"learning_rate": learning_rate,
              "max_depth": depth,
              "n_estimators": estimators,
              "gamma": gamma,
              "min_child_weight": child_weight,
              "subsample": subsample}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    summary = nested_cross_validate(X=X,
                                    Y=Y,
                                    estimator=xgboost, 
                                    c_grid=c_grid,
                                    seed=seed)
    return summary



#### Random Forest ####
def RF(X,
       Y,
       depth=None, 
       estimators=None, 
       impurity=None,
       seed=None):

    ### model
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    
    ### parameters
    c_grid = {"n_estimators": estimators,
              "max_depth": depth,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    summary = nested_cross_validate(X=X,
                                    Y=Y,
                                    estimator=rf,
                                    c_grid=c_grid,
                                    seed=seed)
    return summary



#### CART ####
def CART(X,
         Y,
         depth=None, 
         split=None, 
         impurity=None,
         seed=None):

    ### model
    cart = DecisionTreeClassifier(random_state=seed)
    
    ### parameters
    c_grid = {"max_depth": depth,
              "min_samples_split": split,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    summary = nested_cross_validate(X=X,
                                    Y=Y,
                                    estimator=cart,
                                    c_grid=c_grid,
                                    seed=seed)
    return summary


#### Linear SVM ###

def LinearSVM(X, 
              Y,
              C, 
              seed=None):
    
    ### model
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    
    ### parameter
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    index = 'svm' ## specific for SVM
    
    summary = nested_cross_validate(X=X,
                                    Y=Y,
                                    estimator=svm,
                                    c_grid=c_grid,
                                    seed=seed, 
                                    index = index)
    return summary

#### L1 Logistic Regression -- Abbreviated as Lasso ####
def Lasso(X, 
          Y,
          C,
          seed=None):
    
    ### model 
    lasso = LogisticRegression(class_weight='balanced',
                               solver='liblinear', 
                               random_state=seed, 
                               penalty = 'l1')
    ### parameter
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    summary = nested_cross_validate(X=X,
                                    Y=Y,
                                    estimator=lasso,
                                    c_grid=c_grid, 
                                    seed=seed)
    return summary

#### L2 Logistic Regression ####
def Logistic(X, 
             Y,
             C,
             seed=None):
    
    ### model
    lr = LogisticRegression(class_weight='balanced',
                            solver='liblinear', 
                            random_state=seed, 
                            penalty = 'l2')
    
    ### parameter
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    summary = nested_cross_validate(X=X, 
                                    Y=Y,
                                    estimator=lr,
                                    c_grid=c_grid,
                                    seed=seed)
    return summary
