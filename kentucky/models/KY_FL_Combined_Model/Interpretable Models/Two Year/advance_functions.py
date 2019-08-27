## GAM -- generalized additive model
def EBM(x_KY, y_KY, x_FL, y_FL, learning_rate, depth, estimators, seed):
    
    import numpy as np
    from sklearn.model_selection import KFold, GridSearchCV
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.metrics import roc_auc_score
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate} 
    
    ### nested cross validation
    clf = GridSearchCV(estimator=gam, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x_KY,y_KY)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    
    gam = ExplainableBoostingClassifier(random_state = seed, 
                                        n_estimators = best_param['n_estimators'], 
                                        max_tree_splits = best_param['max_tree_splits'], 
                                        learning_rate = best_param['learning_rate']).fit(x_KY, y_KY)
    FL_score = roc_auc_score(y_FL, gam.predict_proba(x_FL)[:,1])
    
    return best_auc, best_std, auc_diff, best_param, FL_score
