from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness
from sklearn.calibration import CalibratedClassifierCV

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



def nested_cross_validate(X, Y, estimator, c_grid, seed, index = None):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    ## 5 sets of train & test index
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    best_params = []
    auc_diffs = []
    holdout_with_attr_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_auc = []
    confusion_matrix_rets = []
    calibrations = []
    
    ## inner cv
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy()
        
        ## remove unused feature in modeling
        train_x = train_x.drop(['person_id', 'screening_date', 'race'], axis=1).values
        test_x = test_x.drop(['person_id', 'screening_date', 'race'], axis=1).values
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring='roc_auc',
                           cv=inner_cv, return_train_score=True).fit(train_x, train_y)

        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
        else:
            best_model = clf.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
        
        ## confusion matrix stats
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                     preds=holdout_pred,
                                                     labels=test_y, protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ## calibration
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=prob, labels=test_y, protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)

        ## store results
        holdout_with_attr_test.append(holdout_with_attrs)
        holdout_probability.append(prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_auc.append(roc_auc_score(test_y, prob))
        best_params.append(best_param)

    df = pd.concat(confusion_matrix_rets, ignore_index=True)
    df.sort_values(["Attribute", "Attribute Value"], inplace=True)
    df = df.reset_index(drop=True)
    
    calibration_df = pd.concat(calibrations, ignore_index=True)
    calibration_df.sort_values(["Attribute", "Lower Limit Score", "Upper Limit Score"], inplace=True)
    calibration_df = calibration_df.reset_index(drop=True)
    
    return {'best_param': best_params,
            'auc_diffs': auc_diffs,
            'holdout_test_auc': holdout_auc,
            'holdout_with_attrs_test': holdout_with_attr_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': df, 
            'calibration_stats': calibration_df}