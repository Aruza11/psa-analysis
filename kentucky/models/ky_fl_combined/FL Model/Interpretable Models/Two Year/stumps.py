
def stump_features(x,y, columns, alpha, seed):
    
    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import roc_curve, auc
    
    
    ### cross validation -- parameter selection
    lasso = Lasso(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"alpha": alpha}
    clf = GridSearchCV(estimator=lasso, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x,y)
    best_param = clf.best_params_
    
    
    ## run model with best parameter
    lasso = Lasso(random_state=816, alpha=best_param['alpha']).fit(x,y)
    coefs = lasso.coef_[lasso.coef_ != 0]
    features = columns[lasso.coef_ != 0].tolist()
    intercept = round(lasso.intercept_,3)
    
    ## dictionary
    lasso_dict_rounding = {}
    for i in range(len(features)):
        lasso_dict_rounding.update({features[i]: round(coefs[i], 3)})
        
    ### second cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=816)
    train_auc = []
    test_auc = []

    i = 0
    for train, test in cv.split(x,y):    
        train_pred = test_pred = 0  
        X_train, Y_train = x.iloc[train], y[train]
        X_test, Y_test = x.iloc[test], y[test]
        
        for k in features:
            train_values = X_train[k]*lasso_dict_rounding[k]
            test_values = X_test[k]*lasso_dict_rounding[k]
            train_pred += train_values
            test_pred += test_values
            
        train_pred += intercept
        test_pred += intercept
        
        ## auc
        train_fpr, train_tpr, train_thresholds = roc_curve(Y_train, train_pred)
        test_fpr, test_tpr, test_thresholds = roc_curve(Y_test, test_pred)
        train_auc.append(auc(train_fpr, train_tpr))
        test_auc.append(auc(test_fpr, test_tpr))
        i += 1
        
    return {'coefs': coefs, 
            'features': features, 
            'intercept': intercept, 
            'dictionary': lasso_dict_rounding, 
            'param': best_param,
            'train_auc': train_auc, 
            'test_auc': test_auc}


def stump_table(coefs, features, intercept, dictionary):
    
    import numpy as np
    
    print('+-----------------------------------+----------------+')
    print('|', 'Features', '{n:>{ind}}'.format(n = '|', ind=26), 'Score', '{n:>{ind}}'.format(n = '|', ind=10))
    print('|====================================================|')
    for i in range(len(dictionary)):
        print('|', features[i], '{n:>{ind}}'.format(n = '|', ind=35 - len('|'+features[i])), dictionary[features[i]], '{n:>{ind}}'.format(n = '|', ind = 15 - len(np.str(dictionary[features[i]]))))
    print('|', 'Intercept', '{n:>{ind}}'.format(n = '|', ind=25), intercept, '{n:>{ind}}'.format(n = '|', ind = 15 - len(np.str(intercept)))) 
    print('|====================================================|')
    print('|', 'ADD POINTS FROM ROWS 1 TO', len(dictionary), 
          '{n:>{ind}}'.format(n = '|', ind = 6), 'Total Score', '{n:>{ind}}'.format(n = '|', ind = 4))
    print('+-----------------------------------+----------------+')
    
    
    
          
def stump_plots(features, coefs):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stump_visulization(label, sub_features, features, coefs):
        cutoffs = []
        cutoff_values = []        
        cutoff_prep = []
        cutoff_values_prep = []
        
        ## select features
        if label == 'age_at_current_charge':
            
            ## sanity check
            if len(sub_features) == 1:
                cutoffs.append(int(sub_features[0][sub_features[0].find('=')+1:]))
                cutoff_values.append(coefs[np.where(np.array(features) == sub_features[0])[0][0]])
                
                ## prepare values
                cutoff_prep.append(np.linspace(18, cutoffs[0]+0.5, 1000))
                cutoff_prep.append(np.linspace(cutoffs[0]+0.5, 70, 1000))
                cutoff_values_prep.append(np.repeat(cutoff_values[0], 1000))
                cutoff_values_prep.append(np.repeat(0, 1000))
                
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #plt.vlines(x=cutoffs[0]+0.5, ymin=0, ymax=cutoff_values[0], colors='C0', linestyles='dashed')
                plt.title(label)
                plt.show()
            else:
                for j in sub_features:
                    cutoff_values.append(coefs[np.where(np.array(features) == j)[0][0]])
                    cutoffs.append(int(j[j.find('=')+1:])) 
                
                ## prepare cutoff values for plots
                for n in range(len(cutoffs)-1):
                    cutoff_prep.append(np.linspace(cutoffs[n]-0.5, cutoffs[n+1]-0.5, 1000))
                    cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[n:]), 1000)) 
                cutoff_prep.append(np.linspace(cutoffs[-1]-0.5, 70, 1000))
                cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[-1]), 1000)) 
                
                ## visulization
                unique = np.unique(cutoff_values_prep)[::-1]
                unique_len = len(unique)
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #for m in range(1,unique_len):
                #    plt.vlines(x=cutoffs[m]-0.5, ymin=unique[m], ymax=unique[m-1], colors = "C0", linestyles='dashed')
                plt.title(label)
                plt.show()
        else:
            ## sanity check
            if len(sub_features) == 1:
                cutoffs.append(int(sub_features[0][sub_features[0].find('=')+1:]))
                cutoff_values.append(coefs[np.where(np.array(features) == sub_features[0])[0][0]])
                
                ## prepare values
                cutoff_prep.append(np.linspace(-0.5, cutoffs[0]-0.5, 1000))
                cutoff_prep.append(np.linspace(cutoffs[0]-0.5, cutoffs[0]+0.5, 1000))
                cutoff_values_prep.append(np.repeat(0, 1000))
                cutoff_values_prep.append(np.repeat(cutoff_values[0], 1000))
                
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #plt.vlines(x=cutoffs[0]-0.5, ymin=0, ymax=cutoff_values[0], colors='C0', linestyles='dashed')
                plt.title(label)
                plt.show()     
            else:
                for j in sub_features:
                    cutoff_values.append(coefs[np.where(np.array(features) == j)[0][0]])
                    cutoffs.append(int(j[j.find('=')+1:])) 
                
                ## prepare cutoff values for plots
                cutoff_prep = []
                cutoff_values_prep = []
                
                for n in range(len(cutoffs)-1):
                    cutoff_prep.append(np.linspace(cutoffs[n]-0.5, cutoffs[n+1]-0.5, 1000))
                    cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[:n+1]), 1000))    
                cutoff_prep.append(np.linspace(cutoffs[-1]-0.5, cutoffs[-1]+0.5, 1000))
                cutoff_values_prep.append(np.repeat(np.sum(cutoff_values), 1000))   
                
                ## visualization
                unique = np.unique(cutoff_values_prep)
                unique_len = len(unique)
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #for m in range(1, unique_len):
                #    plt.vlines(x=cutoffs[m]-0.5, ymin=unique[m], ymax=unique[m-1], colors = "C0", linestyles='dashed')
                plt.title(label)
                plt.show()  
                
    
    labels = ['Gender', 'age_at_current_charge', 'arrest', 'charges', 'violence', 'felony', 'misdemeanor', 'property', 'murder', 
          'assault', 'sex_offense', 'weapon', 'felprop_viol', 'felassault', 'misdeassult', 'traffic', 'drug', 'dui', 
          'stalking', 'voyeurism', 'fraud', 'stealing', 'trespass', 'ADE', 'Treatment', 'prison', 'jail', 'fta_two_year', 
          'fta_two_year_plus', 'pending_charge', 'probation', 'SentMonths', 'six_month', 'one_year', 'three_year', 
          'five_year', 'current_violence']
    
    for i in labels:
        sub_features = np.array(np.array(features)[[i in k for k in features]])
        if len(sub_features) == 0:
            continue
        stump_visulization(i, sub_features, features, coefs)
    
 
    