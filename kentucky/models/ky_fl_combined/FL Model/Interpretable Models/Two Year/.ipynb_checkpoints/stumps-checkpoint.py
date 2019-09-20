import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score


def stump_cv(KY_x, KY_y,FL_x, FL_y, columns, c_grid, seed):
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    cross_validation = KFold(n_splits=5, random_state=seed, shuffle=True)
    clf = GridSearchCV(estimator=lasso, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(FL_x, FL_y)

    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### best model
    best_model = LogisticRegression(class_weight = 'balanced', 
                                    solver='liblinear', 
                                    random_state=seed, 
                                    penalty='l1', 
                                    C=best_param['C']).fit(FL_x, FL_y)
    coefs = best_model.coef_[best_model.coef_ != 0]
    features = columns[best_model.coef_[0] != 0].tolist()
    intercept = round(best_model.intercept_[0],3)
       
    ## dictionary
    lasso_dict_rounding = {}
    for i in range(len(features)):
        lasso_dict_rounding.update({features[i]: round(coefs[i], 3)})
        
    ## prediction on test set
    prob = 0
    for k in features:
        test_values = KY_x[k]*(lasso_dict_rounding[k])
        prob += test_values
    holdout_prob = np.exp(prob)/(1+np.exp(prob))
    KY_score = roc_auc_score(KY_y, holdout_prob)
        
    return {'best_auc': best_auc,
            'best_params': best_param,
            'auc_diffs': auc_diff,
            'KY_score': KY_score}


def latex_stump_table(coefs, features, intercept, dictionary):
    print('\begin{tabular}{|l|r|r|} \hline')
    for i in range(len(dictionary)):
        sign = '+' if dictionary[features[i]] >= 0 else '-'
        print('{index}.'.format(index = i+1), features[i], '&',np.abs(dictionary[features[i]]), '&', sign+'...', '\\ \hline')
    print('{}.'.format(len(dictionary)+1), 'Intercept', '&', round(intercept, 3), '&', sign+'...', '\\ \hline')
    print('\textbf{ADD POINTS FROM ROWS 1 TO {length}}  &  \textbf{SCORE} & = ..... \\ \hline'
              .format(length=len(dictionary)+1))
    print('\multicolumn{3}{l}{Pr(Y = 1) = exp(score/100) / (1 + exp(score/100))} \\ \hline')
    
          
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
                plt.ylabel('probability')
                plt.show()
            else:
                for j in sub_features:
                    cutoff_values.append(coefs[np.where(np.array(features) == j)[0][0]])
                    cutoffs.append(int(j[j.find('=')+1:])) 
                
                cutoffs.insert(0,18)
                cutoffs.append(70)
                cutoff_values.append(0)
                
                ## prepare cutoff values for plots
                for n in range(len(cutoffs)-1):
                    cutoff_prep.append(np.linspace(cutoffs[n]+0.5, cutoffs[n+1]+0.5, 1000))
                    cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[n:]), 1000)) 
                    
                ## visulization
                unique = np.unique(cutoff_values_prep)[::-1]
                unique_len = len(unique)
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #for m in range(1,unique_len):
                #    plt.vlines(x=cutoffs[m]-0.5, ymin=unique[m], ymax=unique[m-1], colors = "C0", linestyles='dashed')
                plt.title(label)
                plt.ylabel('probability')
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
                plt.ylabel('probability')
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
                plt.ylabel('probability')
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
    
 
    