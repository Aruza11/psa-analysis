### XGBoost
def XGB(x_KY, y_KY, x_FL, y_FL, learning_rate, depth, estimators, seed):
    
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"learning_rate": learning_rate, "max_depth": depth, "n_estimators": estimators}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=xgboost, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    xgboost = xgb.XGBClassifier(random_state=seed, 
                                learning_rate=best_param['learning_rate'], 
                                max_depth = best_param['max_depth'], 
                                n_estimators = best_param['n_estimators']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, xgboost.predict_proba(x_FL)[:,1])
    return best_auc, best_std, auc_diff, best_param, FL_score



### Random Forest
def RF(x_KY, y_KY, x_FL, y_FL, depth, estimators, seed):
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, "max_depth": depth}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=rf, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    rf = RandomForestClassifier(bootstrap=True, random_state=seed, 
                                n_estimators=best_param['n_estimators'], max_depth=best_param['max_depth']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, rf.predict_proba(x_FL)[:,1])
    return best_auc, best_std, auc_diff, best_param, FL_score



### CART
def CART(x_KY, y_KY, x_FL, y_FL, depth, seed):
    
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"max_depth": depth}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=cart, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    
    cart = DecisionTreeClassifier(random_state=seed, max_depth=best_param['max_depth']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, cart.predict_proba(x_FL)[:,1])
    return best_auc, best_std, auc_diff, best_param, FL_score



### Linear SVM
def LinearSVM(x_KY, y_KY, x_FL, y_FL, C, seed):

    import numpy as np
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score

    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=svm, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed, C=best_param['C']).fit(x_KY, y_KY)
    
    y_FL = y_FL.reshape(-1,1)
    prob = (np.array(svm.coef_@x_FL.T) + svm.intercept_).reshape(-1,1)
    FL_score = roc_auc_score(y_FL, prob)
    return best_auc, best_std, auc_diff, best_param, FL_score


### Lasso
def Lasso(x_KY, y_KY, x_FL, y_FL, alpha, seed):

    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    lasso = Lasso(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"alpha": alpha}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lasso, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    lasso = Lasso(random_state=seed, alpha = best_param['alpha']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, lasso.predict(x_FL))
    
    return best_auc, best_std, auc_diff, best_param, FL_score



### Logistic
def Logistic(x_KY, y_KY, x_FL, y_FL, C, seed):
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### cross validation
    clf = GridSearchCV(estimator=lr, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### use best parameter to build model
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, C = best_param['C']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, lr.predict_proba(x_FL)[:,1])
    
    return best_auc, best_std, auc_diff, best_param, FL_score
