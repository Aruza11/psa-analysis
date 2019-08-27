### XGBoost
def XGB(train_x, train_y, test_x, test_y, learning_rate, depth, estimators, seed):
    
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values    
    
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"learning_rate": learning_rate, "max_depth": depth, "n_estimators": estimators}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=xgboost, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x,train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test set
    xgboost = xgb.XGBClassifier(random_state=seed, learning_rate = best_param['learning_rate'], 
                                max_depth = best_param['max_depth'], n_estimators = best_param['n_estimators']).fit(train_x, train_y)
    holdout_prob = xgboost.predict_proba(test_x)[:,1]
    holdout_pred = xgboost.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob) 
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 


### Random Forest
def RF(train_x, train_y, test_x, test_y, depth, estimators, seed):
    
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values    
    
    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, "max_depth": depth}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=rf, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x, train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test set
    rf = RandomForestClassifier(bootstrap=True, random_state=seed, 
                                n_estimators=best_param['n_estimators'], 
                                max_depth = best_param['max_depth']).fit(train_x, train_y)
    holdout_prob = rf.predict_proba(test_x)[:,1]
    holdout_pred = rf.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob) 
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 



### CART
def CART(train_x, train_y, test_x, test_y, depth, seed):
    
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values    
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"max_depth": depth}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=cart, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x, train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test set
    cart = DecisionTreeClassifier(random_state=seed, max_depth = best_param['max_depth']).fit(train_x, train_y)
    holdout_prob = cart.predict_proba(test_x)[:,1]
    holdout_pred = cart.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)    
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 




### Linear SVM
def LinearSVM(train_x, train_y, test_x, test_y, C, seed):
    
    import numpy as np
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values
    
    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=svm, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x,train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test set
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed, C=best_param['C']).fit(train_x, train_y)
    holdout_prob = (svm.coef_@test_x.T + svm.intercept_).reshape(-1,1)
    holdout_pred = svm.predict(test_x)
    test_y = test_y.reshape(-1,1)
    holdout_auc = roc_auc_score(test_y, holdout_prob)    
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 


### Lasso
def Lasso(train_x, train_y, test_x, test_y, alpha, seed):

    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
        
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values
    
    ### model & parameters
    lasso = Lasso(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"alpha": alpha}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lasso, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x,train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test
    lasso = Lasso(random_state = seed, alpha = best_param['alpha']).fit(train_x, train_y)
    holdout_prob = lasso.predict(test_x)
    holdout_pred = (holdout_prob > 0.5)
    holdout_auc = roc_auc_score(test_y, holdout_prob)
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 



### Logistic
def Logistic(train_x, train_y, test_x, test_y, C, seed):
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"C": C}
    
    ### cross validation
    clf = GridSearchCV(estimator=lr, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x, train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### holdout test
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, C=best_param['C']).fit(train_x, train_y)
    holdout_prob = lr.predict_proba(test_x)[:,1]
    holdout_pred = lr.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob)
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 
    
