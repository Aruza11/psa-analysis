from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_fairness



def cross_validate(X, Y, estimator, c_grid, seed):
    """Performs cross validation and selects a model given X and Y dataframes, 
    an estimator, a dictionary of parameters, and a random seed. 
    """
    # settings 
    n_splits = 5
    scoring = 'roc_auc'

    cross_validation = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring=scoring,
                       cv=cross_validation, return_train_score=True).fit(X, Y)
    mean_train_score = clf.cv_results_['mean_train_score']
    mean_test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']

    # scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(mean_test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = mean_train_score[np.where(mean_test_score == clf.best_score_)[
        0][0]] - clf.best_score_

    return mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff

def nested_cross_validate(X, Y, estimator, c_grid, seed, holdout_with_attrs):
    train_outer = []
    test_outer = []
    outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)

    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)

    holdout_auc = []
    best_params = []
    auc_diffs = []
    fairness_overviews = []

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        train_x, test_x = X[train_outer[i]], X[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]


        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring='roc_auc',
                           cv=inner_cv, return_train_score=True).fit(train_x, train_y)

        ## best parameter & scores
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        best_param = clf.best_params_
        auc_diffs.append(np.mean(train_score) - np.mean(test_score))

        ## train model on best param
        best_model = clf.fit(train_x, train_y)
        prob = best_model.predict_proba(test_x)[:, 1]
        holdout_pred = best_model.predict(test_x)

        holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                     preds=holdout_pred,
                                                     labels=test_y)
        fairness_overviews.append(holdout_fairness_overview)

        ## store results
        holdout_auc.append(roc_auc_score(test_y, prob))
        best_params.append(best_param)

    return np.mean(holdout_auc), np.std(holdout_auc), best_params, auc_diffs, fairness_overviews
