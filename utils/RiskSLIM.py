import numpy as np
import pandas as pd

from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa



def risk_slim(data, max_coefficient, max_L0_value, c0_value, max_runtime = 120, w_pos = 1, max_offset=50):
    
    """
    @parameters:
    
    max_coefficient:  value of largest/smallest coefficient
    max_L0_value:     maximum model size (set as float(inf))
    max_offset:       maximum value of offset parameter (optional)
    c0_value:         L0-penalty parameter such that c0_value > 0; larger values -> 
                      sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    max_runtime:      max algorithm running time
    w_pos:            relative weight on examples with y = +1; w_neg = 1.00 (optional)
    
    """
    
    # create coefficient set and set the value of the offset parameter
    coef_set = CoefficientSet(variable_names = data['variable_names'], lb = 0, ub = max_coefficient, sign = 0)
    conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
    max_offset = min(max_offset, conservative_offset)
    coef_set['(Intercept)'].ub = max_offset
    coef_set['(Intercept)'].lb = -max_offset

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # Set parameters
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        'w_pos': w_pos,

        # LCPA Settings
        'max_runtime': max_runtime,                         # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
        
        # LCPA Improvements
        'round_flag': False,                                # round continuous solutions with SeqRd
        'polish_flag': False,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': False,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        
        # Initialization
        'initialization_flag': True,                        # use initialization procedure
        'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,

        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)
        
    return model_info, mip_info, lcpa_info




def riskslim_prediction(X, feature_name, model_info):
    
    """
    @parameters
    
    X: test input features (np.array)
    feature_name: feature names
    model_info: output from RiskSLIM model
    
    """
    
    ## initialize parameters
    dictionary = {}
    prob = np.zeros(len(X))
    scores = np.zeros(len(X))
    
    ## prepare statistics
    subtraction_score = model_info['solution'][0]
    coefs = model_info['solution'][1:]
    index = np.where(coefs != 0)[0]
    
    nonzero_coefs = coefs[index]
    features = feature_name[index]
    X_sub = X[:,index]
    
    ## build dictionaries
    for i in range(len(features)):
        single_feature = features[i]
        coef = nonzero_coefs[i]
        dictionary.update({single_feature: coef})
        
    ## calculate probability
    for i in range(len(X_sub)):
        summation = 0
        for j in range(len(features)):
            a = X_sub[i,j]
            summation += dictionary[features[j]] * a
        scores[i] = summation
    
    prob = 1/(1+np.exp(-(scores + subtraction_score)))
    
    return prob



def riskslim_accuracy(X, Y, feature_name, model_info, threshold=0.5):
    
    prob = riskslim_prediction(X, feature_name, model_info)
    pred = np.mean((prob > threshold) == Y)
    
    return pred



def risk_cv(X, 
            Y,
            indicator,
            y_label, 
            max_coef,
            max_coef_number,
            max_runtime,
            c,
            seed):

    ## set up data
    Y = Y.reshape(-1,1)
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    train_auc = []
    test_auc = []
    holdout_with_attrs_test = []
    holdout_probability = []
    holdout_prediction = []
    holdout_y = []
    confusion_matrix_rets = []
    calibrations = []
    
    i = 0
    for train, test in cv.split(X, Y):
    
        ## subset train data & store test data
        train_x, train_y = X.iloc[train], Y[train]
        test_x, test_y = X.iloc[test], Y[test]
        sample_weights_train, sample_weights_test = sample_weights[train], sample_weights[test]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy().drop(['(Intercept)'], axis=1)
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex>=1': 'sex'})
        
        ## remove unused feature in modeling
        if indicator == 1:
            train_x = train_x.drop(['person_id', 'screening_date', 'race'], axis=1)
            test_x = test_x.drop(['person_id', 'screening_date', 'race'], axis=1).values
        else:
            train_x = train_x.drop(['person_id', 'screening_date', 'race', 'sex>=1'], axis=1)
            test_x = test_x.drop(['person_id', 'screening_date', 'race', 'sex>=1'], axis=1).values

        cols = train_x.columns.tolist()
        train_x = train_x.values
        
        ## create new data dictionary
        new_train_data = {
            'X': train_x,
            'Y': train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': sample_weights_train
        }
            
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                    max_coefficient=max_coef, 
                                                    max_L0_value=max_coef_number, 
                                                    c0_value=c, 
                                                    max_runtime=max_runtime)
        print_model(model_info['solution'], new_train_data)
        
        ## change data format
        train_x, test_x = train_x[:,1:], test_x[:,1:] ## remove the first column, which is "intercept"
        train_y[train_y == -1] = 0 ## change -1 to 0
        test_y[test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        train_prob = riskslim_prediction(train_x, np.array(cols), model_info).reshape(-1,1)
        test_prob = riskslim_prediction(test_x, np.array(cols), model_info).reshape(-1,1)
        test_pred = (test_prob > 0.5)
        
        ## AUC
        train_auc.append(roc_auc_score(train_y, train_prob))
        test_auc.append(roc_auc_score(test_y, test_prob))
        
        ## confusion matrix stats
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds=test_pred,
                                                                   labels=test_y, 
                                                                   protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ## calibration
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=test_prob, labels=test_y, protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)
        
        ## store results
        holdout_with_attrs_test.append(holdout_with_attrs_test)
        holdout_probability.append(test_prob)
        holdout_prediction.append(test_pred)
        holdout_y.append(test_y)
        i += 1
        
    df = pd.concat(confusion_matrix_rets, ignore_index=True)
    df.sort_values(["Attribute", "Attribute Value"], inplace=True)    
    df = df.reset_index(drop=True)
         
    calibration_df = pd.concat(calibrations, ignore_index=True)
    calibration_df.sort_values(["Attribute", "Lower Limit Score", "Upper Limit Score"], inplace=True)
    calibration_df = calibration_df.reset_index(drop=True)
    
    return {'train_auc': train_auc, 
            'test_auc': test_auc, 
            'holdout_with_attrs_test': holdout_with_attrs_test,
            'holdout_probability': holdout_probability,
            'holdout_prediction': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': df, 
            'calibration_stats': calibration_df}

    