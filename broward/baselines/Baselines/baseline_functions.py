### XGBoost
def XGB(x, y, learning_rate, depth, estimators, gamma, child_weight, subsample, seed):
    
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    xgb = xgb.XGBClassifier(random_state=seed)
    c_grid = {"learning_rate": learning_rate, 
              "max_depth": depth, 
              "n_estimators": estimators, 
              "gamma": gamma, 
              "min_child_weight": child_weight, 
              "subsample": subsample}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=xgb, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score


### Random Forest
def RF(x, y, depth, estimators, impurity, seed):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_depth": depth, 
              "min_impurity_decrease": impurity}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=rf, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score



### CART
def CART(x, y, depth, split, impurity, seed):
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    c_grid = {"max_depth": depth, 
              "min_samples_split": split, 
              "min_impurity_decrease": impurity}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=cart, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    
    return train_score, test_score



### Linear SVM
def LinearSVM(x, y, C, seed):

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed) 

    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=1e7, random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=svm, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score


### Lasso
def Lasso(x, y, alpha,seed):

    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, KFold, cross_validate

    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    lasso = Lasso(random_state=seed)
    c_grid = {"alpha": alpha}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lasso, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    return train_score, test_score


### Logistic

def Logistic(x, y, C,seed):
  
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, GridSearchCV, cross_validate
    
    ## cross validation set up
    inner_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    outer_cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    c_grid = {"C": C}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=lr, param_grid=c_grid, scoring='roc_auc',cv=inner_cv, return_train_score=True)
    nested_score = cross_validate(clf, X=x, y=y, scoring='roc_auc', cv=outer_cv, return_train_score=True)
    train_score, test_score = nested_score['train_score'], nested_score['test_score']
    
    return train_score, test_score
