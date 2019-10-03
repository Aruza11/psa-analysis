from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from together_utils.fairness_functions import compute_fairness
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def nested_cross_validate(X, Y, indicator, c_grid, seed, index = None):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    ## 5 sets of train & test index
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    combined_holdout_auc = []
    ky_holdout_auc = []
    fl_holdout_auc = []
    best_params = []
    auc_diffs = []
    fairness_overviews = []

    ## inner cv
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        
        ## the whole train and test set
        combined_train_x, combined_train_y = X.iloc[train_outer[i]], Y[train_outer[i]]
        combined_test_x, combined_test_y = X.iloc[test_outer[i]], Y[test_outer[i]]
        
        ## get weights
        weights = np.repeat(1, combined_train_x.shape[0])
        weights[ combined_train_x['index'] == 'FL' ] = 50
        
        ## kentucky test
        ky_test_x = combined_test_x[combined_test_x['index'] == 'KY']
        ky_test_y = combined_test_y[combined_test_x['index'] == 'KY']
        
        ## broward test
        fl_test_x = combined_test_x[combined_test_x['index'] == 'FL']
        fl_test_y = combined_test_y[combined_test_x['index'] == 'FL']

        ## remove unused feature in modeling
        combined_train_x = combined_train_x.drop(['person_id', 'screening_date', 'race', 'index'], axis=1)
        combined_test_x = combined_test_x.drop(['person_id', 'screening_date', 'race', 'index'], axis=1)
        ky_test_x = ky_test_x.drop(['person_id', 'screening_date', 'race', 'index'], axis=1)
        fl_test_x = fl_test_x.drop(['person_id', 'screening_date', 'race', 'index'], axis=1)
        
        
        ## get estimator
        if indicator == "logistic":
            estimator = LogisticRegression(class_weight=weights,
                                           solver='liblinear', 
                                           random_state=seed, 
                                           penalty = 'l2')
        elif: indicator == "lasso":
            estimator = LogisticRegression(class_weight=weights,
                                           solver='liblinear', 
                                           random_state=seed, 
                                           penalty = 'l1')
        elif: indicator == "svm":
            estimator = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
        elif: indicator == "rf":
            estimator = RandomForestClassifier(bootstrap=True, random_state=seed)
        else: 
            estimator = xgb.XGBClassifier(random_state=seed)
        
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring='roc_auc',
                           cv=inner_cv, return_train_score=True).fit(combined_train_x, combined_train_y)

        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(combined_train_x, combined_train_y)
            combined_prob = best_model.predict_proba(combined_test_x)[:, 1]
            ky_prob = best_model.predict_proba(ky_test_x)[:, 1]
            fl_prob = best_model.predict_proba(fl_test_x)[:, 1]
            
        else:
            best_model = clf.fit(combined_train_x, combined_train_y)
            combined_prob = best_model.predict_proba(combined_test_x)[:, 1]
            ky_prob = best_model.predict_proba(ky_test_x)[:, 1]
            fl_prob = best_model.predict_proba(fl_test_x)[:, 1]

        ## store results
        combined_holdout_auc.append(roc_auc_score(combined_test_y, combined_prob))
        ky_holdout_auc.append(roc_auc_score(ky_test_y, ky_prob))
        fl_holdout_auc.append(roc_auc_score(fl_test_y, fl_prob))
        best_params.append(best_param)

    return combined_holdout_auc, ky_holdout_auc, fl_holdout_auc, best_params, auc_diffs, fairness_overviews
