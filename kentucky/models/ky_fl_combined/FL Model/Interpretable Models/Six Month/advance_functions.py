### AdaBoost -- one-depth decision tree
def Adaboost(x, y, learning_rate, estimators, seed):
    
    import numpy as np
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.ensemble import AdaBoostClassifier
    
    ### model & parameters
    ada = AdaBoostClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "learning_rate": learning_rate}  
    
    ### nested cross validation
    clf = GridSearchCV(estimator=ada, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x,y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    return best_auc, best_std, auc_diff, best_param



## GAM -- generalized additive model
def EBM(x, y, learning_rate, depth, estimators, seed):
    
    import numpy as np
    from sklearn.model_selection import KFold, GridSearchCV
    from interpret.glassbox import ExplainableBoostingClassifier
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate} 
    
    ### nested cross validation
    clf = GridSearchCV(estimator=gam, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x,y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    return best_auc, best_std, auc_diff, best_param
