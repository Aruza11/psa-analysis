import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


### XGBoost
def XGB(KY_x, KY_y, FL_x, FL_y, learning_rate, depth, estimators, gamma, child_weight, subsample, seed):
    
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"learning_rate": learning_rate, 
              "max_depth": depth, 
              "n_estimators": estimators, 
              "gamma": gamma, 
              "min_child_weight": child_weight, 
              "subsample": subsample}
    
    ### cross validation
    clf = GridSearchCV(estimator=xgboost, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    xgboost = clf.fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, xgboost.predict_proba(KY_x)[:,1])
    
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}



### Random Forest
def RF(KY_x, KY_y, FL_x, FL_y, depth, estimators,impurity, seed):

    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_depth": depth, 
              "min_impurity_decrease": impurity}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=rf, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    rf = clf.fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, rf.predict_proba(KY_x)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}



### CART
def CART(KY_x, KY_y, FL_x, FL_y, depth, split, impurity, seed):
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"max_depth": depth, 
              "min_samples_split": split, 
              "min_impurity_decrease": impurity}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=cart, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    cart = clf.fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, cart.predict_proba(KY_x)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}



### Linear SVM
def LinearSVM(KY_x, KY_y, FL_x, FL_y, C, seed):

    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=svm, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    svm = CalibratedClassifierCV(clf, cv=5).fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, svm.predict_proba(KY_x)[:, 1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}


### Lasso
def Lasso(KY_x, KY_y, FL_x, FL_y, C, seed):
    
    ### model & parameters
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lasso, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    lasso = clf.fit(FL_x,FL_y)
    KY_score = roc_auc_score(KY_y, lasso.predict_proba(KY_x)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}



### Logistic
def Logistic(KY_x, KY_y, FL_x, FL_y, C, seed):
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### cross validation
    clf = GridSearchCV(estimator=lr, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### use best parameter to build model
    lr = clf.fit(FL_x,FL_y)
    KY_score = roc_auc_score(KY_y, lr.predict_proba(KY_x)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}
