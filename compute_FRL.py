# run using python -m main (no .py)

import sys,os
srcpath = "C:/Users/Caroline Wang/src/FRL_optimization"
sys.path.insert(0, srcpath)

import pandas as pd 
import numpy as np

from utils import expand_grid
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import matplotlib.pyplot as plt

#load FRL code
from data import load_data, load_labels
from antecedents import mine_antecedents
from FRL import learn_FRL
from softFRL import learn_softFRL
from display import display_rule_list, display_softFRL

def fit_FRL(data, params):
    '''
    @param data: dictionary containing training data as 'X' and labels as 'Y'(both ndarrays) 
                and a list of var names
    @param params = {
            "w" : [7],   
            "C" : [0.000001],                                         
            "prob_terminate" : [0.01],                                            
            "T_FRL" : [3000],    
            "lmda": [0.8]                                                           
        }


    '''
    #LEARN FRL
    # mine rules
    print("mining rules using FP-growth")
    minsupport = 10
    max_predicates_per_ant = 2
    X_pos,X_neg,nantecedents,antecedent_len,antecedent_set = \
        mine_antecedents(data['X'],data['Y'],minsupport,max_predicates_per_ant)
    
    n = len(data['X'])
    
    # learn a falling rule list from the training data
    # train a falling rule list
    print("running algorithm FRL on bank-full")
    FRL_rule, FRL_prob, FRL_pos_cnt, FRL_neg_cnt, FRL_obj_per_rule, FRL_Ld, \
        FRL_Ld_over_iters, FRL_Ld_best_over_iters = \
        learn_FRL(X_pos, X_neg, n, params['w'], params['C'], params['prob_terminate'], int(params['T_FRL']), params['lmda'])
    
    print("FRL learned:") #PROBLEM: NO CANDIDATE ANTECEDENTS FOUND
    display_rule_list(FRL_rule, FRL_prob, antecedent_set, FRL_pos_cnt, FRL_neg_cnt,
                      FRL_obj_per_rule, FRL_Ld)

    def rule_list(x):
        '''
        Using the computed FRL, evaluate on a single entry x (a list)
        '''
        #FRL_rule, FRL_prob, antecedent_set
        #return true if any of X's features == the rule
        rule_true = lambda rule: any(x_item == rule for x_item in x)

        for i in range(len(FRL_rule) - 1):
            # sometimes rules are multipart so 
            # we check if all parts are true
            if all(rule_true(rule) for rule in antecedent_set[FRL_rule[i]]): 
                return FRL_prob[i]
        #none of rules held 
        return FRL_prob[-1]

    return rule_list

def predict(X,rule_list, threshold = None):
    '''
    @param X: a list of lists
    returns np array of predictions for X; probabilities if threshold == None, 
    predictions if threshold != None (works with binary class. only)
    '''
    preds = []
    for x in X: 
        if threshold == None: 
            preds.append(rule_list(x))
        else: 
            preds.append(0 if rule_list(x) < threshold else 1)
    return np.array(preds)


if __name__ == '__main__':
    train_name = "\\bin_train"
    test_name = "\\bin_test"                              
   

    data_dir = os.getcwd()                                      # directory where datasets are stored
    train_csv_file = data_dir + train_name + '_data.csv'          # csv file for the dataset
    test_csv_file = data_dir + test_name + '_data.csv'            # held out test set


    train_df, test_df = pd.read_csv(train_csv_file, sep=','), pd.read_csv(test_csv_file, sep=',')
    subset = ["recid_use",
      "p_current_age18", "p_current_age1929", "p_current_age3039", "p_current_age4049", "p_current_age5059", "p_current_age60plus",
      "p_property0","p_property13", "p_property46", "p_property79", "p_property10up", 
      "prior_conviction_M01", "prior_conviction_M24", "prior_conviction_M57", "prior_conviction_M810" , "prior_conviction_M11up",
      "p_charge0", "p_charge13", "p_charge45", "p_charge67", "p_charge810" , "p_charge11up", 
      "p_felprop_violarrest0", "p_felprop_violarrest1", "p_felprop_violarrest2", "p_felprop_violarrest3", "p_felprop_violarrest46", "p_felprop_violarrest7up",  
      "total_convictions0", "total_convictions1", "total_convictions2", "total_convictions3", "total_convictions46", "total_convictions7up"]
    train_df, test_df = train_df[subset], test_df[subset]
    variable_names = list(train_df)[1:] #do not want column name outcome label
    # print(variable_names)
    # sys.exit(0)
    #remove label column and headers from data
    X_train, X_test = train_df.iloc[1:, 1:], test_df.iloc[1:, 1:]
    X_train, X_test = X_train.values.tolist(), X_test.values.tolist()

    #reformat
    f = lambda x , y,data : variable_names[x] + "=" + str(data[y][x] )
    for i in range(len(X_train)): #each observation
        for j in range(len(variable_names)): #each datapoint
            X_train[i][j] = f(j, i, X_train)

    for i in range(len(X_test)): #each observation
        for j in range(len(variable_names)): #each datapoint
            X_test[i][j] = f(j, i, X_test)

    #a numpy array
    #FRL needs y in {0,1}
    Y_train =  np.array(train_df.iloc[1:,0]) 
    Y_train[Y_train == -1] = 0

    Y_test =  np.array(test_df.iloc[1:,0]) 
    Y_test[Y_test == -1] = 0
                
    # problem parameters
    params = {
        "w" : [7],   
        "C" : [0.000001],                                         
        "prob_terminate" : [0.01],                                            
        "T_FRL" : [3000],    
        "lmda": [0.8]                               #curiosity function                                    
    }

    setup = {
        "nfolds" : 5
    }

    param_grid = expand_grid(params) #list of dictionaries 
    nparam = len(param_grid)
    eval_varnames = dict.fromkeys(["train_auc_mean","train_auc_std","test_auc_mean","test_auc_std"])
    seeds = np.random.randint(10000, size = nparam)
    performance = [] #a list of dictionaries 

    for i, param_dict in enumerate(param_grid): 
        kfold = KFold(setup["nfolds"], shuffle = True, random_state = seeds[i])
        performance.append({                        #performance dictionaries appended in order
            "i_param": i,                           #i_param should equal index of dict in list
            "seed": seeds[i]}.update(eval_varnames)
            )

        train_auc, test_auc = [],[]
        for train_ind, test_ind in kfold.split(X_train, Y_train):
            train_fold = {
                "X": [X_train[i] for i in train_ind],  #features
                "Y": Y_train[train_ind],                   #labels
                "variable_names" : variable_names
            }

            test_fold = {
                "X": [X_train[i] for i in test_ind],  #features
                "Y": Y_train[test_ind],                   #labels                "variable_names" : variable_names
                "variable_names" : variable_names
            }

            FRL_clf = fit_FRL(train_fold , param_dict)

            train_preds = predict(train_fold['X'],FRL_clf)
            test_preds = predict(test_fold['X'],FRL_clf)
            print(test_preds)
            train_auc.append(roc_auc_score(y_true = train_fold["Y"], y_score = train_preds))
            test_auc.append(roc_auc_score(y_true = test_fold["Y"], y_score = test_preds))

        print(train_auc, test_auc)
        performance[-1]['train_auc_mean'], performance[-1]['train_auc_std'] = np.mean(train_auc), np.std(train_auc) 
        performance[-1]['test_auc_mean'], performance[-1]['test_auc_std'] = np.mean(test_auc), np.std(test_auc) 

    best_performance = max(performance, key=lambda x:x['test_auc_mean'])
    i_param_best = best_performance['i_param']

    #retrain on whole training set using best params, test on test set 
    train_all = {
        "X": X_train,                   #features
        "Y": Y_train,                   #labels
        "variable_names" : variable_names
    }
    test_heldout = {
        "X": X_test,                   #features
        "Y": Y_test,                   #labels
        "variable_names" : variable_names
    }

    best_FRL = fit_FRL(train_all, param_grid[i_param_best])

    test_preds = predict(test_heldout['X'],best_FRL)
    test_auc = roc_auc_score(y_true = test_heldout["Y"], y_score = test_preds)

    print("FRL AUC is ", test_auc)
    skplt.metrics.plot_roc_curve(test_heldout["Y"], test_preds)
    plt.show()

#########################################################
    # # set the parameters of Algorithm softFRL
    # C1 = 0.5
    # T_softFRL = 6000

    # print("running algorithm softFRL on bank-full")
    # softFRL_rule, softFRL_prob, softFRL_pos_cnt, softFRL_neg_cnt, \
    #     softFRL_pos_prop, softFRL_obj_per_rule, softFRL_Ld, \
    #     softFRL_Ld_over_iters, softFRL_Ld_best_over_iters = \
    #     learn_softFRL(X_pos, X_neg, n, w, C, C1, prob_terminate,
    #                   T_softFRL, lmda)
    
    # print("softFRL learned:")    
    # display_softFRL(softFRL_rule, softFRL_prob, antecedent_set,
    #                 softFRL_pos_cnt, softFRL_neg_cnt, softFRL_pos_prop,
    #                 softFRL_obj_per_rule, softFRL_Ld)             

