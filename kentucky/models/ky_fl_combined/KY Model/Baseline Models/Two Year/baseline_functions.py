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
def XGB(x_KY, y_KY, x_FL, y_FL, learning_rate, depth, estimators, seed):
    
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"learning_rate": learning_rate, 
              "max_depth": depth, 
              "n_estimators": estimators}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=xgboost, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(x_KY,y_KY)
    
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    xgboost = clf.fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, xgboost.predict_proba(x_FL)[:,1])
    
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_score':FL_score}



### Random Forest
def RF(x_KY, y_KY, x_FL, y_FL, depth, estimators, seed):

    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_depth": depth}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=rf, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    rf = clf.fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, rf.predict_proba(x_FL)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_score':FL_score}


### Linear SVM
def LinearSVM(x_KY, y_KY, x_FL, y_FL, C, seed):

    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=svm, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    svm = CalibratedClassifierCV(clf, cv=5).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, svm.predict_proba(x_FL)[:, 1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_score':FL_score}


### Lasso
def Lasso(x_KY, y_KY, x_FL, y_FL, C, seed):
    
    ### model & parameters
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lasso, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    lasso = clf.fit(x_KY,y_KY)
    FL_score = roc_auc_score(y_FL, lasso.predict_proba(x_FL)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_score':FL_score}



### Logistic
def Logistic(x_KY, y_KY, x_FL, y_FL, C, seed):
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### cross validation
    clf = GridSearchCV(estimator=lr, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### use best parameter to build model
    lr = clf.fit(x_KY,y_KY)
    FL_score = roc_auc_score(y_FL, lr.predict_proba(x_FL)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_score':FL_score}
