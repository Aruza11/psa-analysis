import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.fairness_functions import compute_fairness
from utils.model_selection import cross_validate


def preprocess(train_x, train_y, test_x, test_y): 
    holdout_with_attrs = test_x.copy()

    train_x = train_x.copy().drop(['person_id', 'screening_date', 'race'], axis=1).values
    test_x = test_x.copy().drop(['person_id', 'screening_date', 'race'], axis=1).values

    train_y = train_y.copy().values
    test_y = test_y.copy().values

    return train_x, train_y, test_x, test_y, holdout_with_attrs


def XGB(train_x, train_y,
        test_x, test_y,
        learning_rate=None, depth=None, estimators=None, gamma=None, child_weight=None, subsample=None,
        seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    c_grid = {"learning_rate": learning_rate,
              "max_depth": depth,
              "n_estimators": estimators,
              "gamma": gamma,
              "min_child_weight": child_weight,
              "subsample": subsample}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                           Y=train_y,
                                                                                                           estimator=xgboost,
                                                                                                           c_grid=c_grid,
                                                                                                           seed=seed)

    # holdout test set
    xgboost = xgb.XGBClassifier(random_state=seed, **best_param).fit(train_x, train_y)
    holdout_prob = xgboost.predict_proba(test_x)[:, 1]
    holdout_pred = xgboost.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}


def RF(train_x, train_y,
       test_x, test_y,
       depth=None, estimators=None, impurity=None,
       seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)

    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    c_grid = {"n_estimators": estimators,
              "max_depth": depth,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                                                                                                                                 Y=train_y,
                                                                                                                                                                                                                 estimator=rf,
                                                                                                                                                                                                                 c_grid=c_grid,
                                                                                                                                                                                                                 seed=seed)

    # holdout test set
    rf = RandomForestClassifier(bootstrap=True, random_state=seed, **best_param).fit(train_x, train_y)
    holdout_prob = rf.predict_proba(test_x)[:, 1]
    holdout_pred = rf.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}


def CART(train_x, train_y,
         test_x, test_y,
         depth=None, split=None, impurity=None,
         seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)

    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    c_grid = {"max_depth": depth,
              "min_samples_split": split,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                           Y=train_y,
                                                                                                           estimator=cart,
                                                                                                           c_grid=c_grid,
                                                                                                           seed=seed)

    # holdout test set
    cart = DecisionTreeClassifier(random_state=seed, **best_param).fit(train_x, train_y)
    holdout_prob = cart.predict_proba(test_x)[:, 1]
    holdout_pred = cart.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}


def LinearSVM(train_x, train_y,
              test_x, test_y,
              C,
              seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)

    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                           Y=train_y,
                                                                                                           estimator=svm,
                                                                                                           c_grid=c_grid,
                                                                                                           seed=seed)
    # holdout test set
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed, **best_param).fit(train_x, train_y)
    holdout_prob = (svm.coef_@test_x.T + svm.intercept_).reshape(-1, 1)
    holdout_pred = svm.predict(test_x)
    test_y = test_y.reshape(-1, 1)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}


def Lasso(train_x, train_y,
          test_x, test_y,
          alpha,
          seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)

    ### model & parameters
    lasso = Lasso_sklearn(random_state=seed)
    c_grid = {"alpha": alpha}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                           Y=train_y,
                                                                                                           estimator=lasso,
                                                                                                           c_grid=c_grid,
                                                                                                           seed=seed)

    # holdout test
    lasso = Lasso_sklearn(random_state=seed, **best_param).fit(
            train_x, train_y)
    holdout_prob = lasso.predict(test_x)
    holdout_pred = (holdout_prob > 0.5)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}


def Logistic(train_x, train_y,
             test_x, test_y,
             C,
             seed=None):

    train_x, train_y, test_x, test_y, holdout_with_attrs = preprocess(train_x, train_y, 
                                                                      test_x, test_y)

    ### model & parameters
    lr = LogisticRegression(class_weight='balanced',
                            solver='liblinear', 
                            random_state=seed)
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff = cross_validate(X=train_x,
                                                                                                           Y=train_y,
                                                                                                           estimator=lr,
                                                                                                           c_grid=c_grid,
                                                                                                           seed=seed)

    # holdout test
    lr = LogisticRegression(class_weight='balanced', 
                            solver='liblinear',
                            random_state=seed, 
                            **best_param).fit(train_x, train_y)

    holdout_prob = lr.predict_proba(test_x)[:, 1]
    holdout_pred = lr.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)

    # compute fairness
    holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                 preds=holdout_pred,
                                                 labels=test_y)

    return {'best_param': best_param,
            'best_validation_auc': best_auc,
            'best_validation_std': best_std,
            'best_validation_auc_diff': auc_diff,
            'holdout_test_auc': holdout_auc,
            'holdout_fairness_overview': holdout_fairness_overview}
