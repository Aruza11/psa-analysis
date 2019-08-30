from sklearn.model_selection import KFold, GridSearchCV
import numpy as np 


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

