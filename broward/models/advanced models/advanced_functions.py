### AdaBoost -- one-depth decision tree
def Adaboost(x, y, learning_rate, estimators, seed):
    
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    ada = AdaBoostClassifier(random_state=816)
    c_grid = {"n_estimators": estimators, 
              "learning_rate": learning_rate}  
    
    ### nested cross validation
    clf = GridSearchCV(estimator=ada, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score


## GAM -- generalized additive model
def GAM(x, y, learning_rate, depth, estimators, holdout_split, seed):
    
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=816)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate, 
              "holdout_split": holdout_split} 
    
    ### nested cross validation
    clf = GridSearchCV(estimator=gam, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score
