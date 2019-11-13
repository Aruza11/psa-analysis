from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd 
import numpy as np
import corels
from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness

def extract_rules(model):
    
    ## extract rules and signs
    rules = model.rl_.rules
    rule_features = model.rl_.features
    
    rule_length = len(rules)
    rule_features_sub = []
    rule_signs = []
    
    for i in range(rule_length-1):
        feature_index = rules[i]['antecedents']
        
        if len(feature_index) == 1:
            rule_features_sub.append( [rule_features[ np.abs(feature_index[0])-1]] )
            rule_signs.append( [np.sign(feature_index[0])] )
        else:
            feature_more_than_one = []
            sign_more_than_one = []
            for j in range(len(feature_index)):
                feature_more_than_one.append( rule_features[ np.abs(feature_index[j])-1 ] )
                sign_more_than_one.append( np.sign(feature_index[j]) )
                
            rule_features_sub.append(feature_more_than_one)
            rule_signs.append(sign_more_than_one)
                    
    return rule_features_sub, rule_signs



def corel_prediction(dataset, rule_features_sub, rule_signs):
    
    ## initialize storage & data
    order, probabilities = [], []
    data = dataset
    
    for i in range(len(rule_features_sub)):
        feature = rule_features_sub[i]
        feature_len = len(feature)
        sign = rule_signs[i]
        
        ## check length of feature
        if feature_len == 1:
            if sign[0] == 1: 
                rule = (data[feature[0]] == 1)
            else: 
                rule = (data[feature[0]] == 0)
        else: 
            ## initialize the first rule
            if sign[0] == 1: 
                rule = (data[feature[0]] == 1)
            else: 
                rule = (data[feature[0]] == 0)
    
            for j in range(feature_len-1):
                if sign[j+1] == 1:
                    rule = (rule & data[feature[j+1]] == 1)
                else:
                    rule = (rule & data[feature[j+1]] == 0)
                    
        ## subset data with the rule
        sub_data = data[rule]
        index = sub_data['index']
        prob = np.sum(sub_data['y'] == 1)/len(sub_data)
        data = data[~rule]
     
        ## save results
        order = order + index.values.tolist()
        probabilities = probabilities + np.repeat(prob, len(index)).tolist()
    
    ## default prediction after the last rule
    index = data['index']
    prob = np.sum(data['y'] == 1)/len(data)
    order = order + index.values.tolist()
    probabilities = probabilities + np.repeat(prob, len(index)).tolist()    
        
    ## prediction table
    prediction = pd.DataFrame(np.c_[order, probabilities], columns=['index', 'probability'])
    prediction_table = pd.merge(dataset, prediction, on='index')
            
    ## prediction and probability
    proba = prediction_table['probability'].values
    pred = (prediction_table['probability'] > 0.5).values
    auc = roc_auc_score(prediction_table['y'], proba)
    
    return proba, pred, auc



def corel_cv(X, Y, max_card, c, seed):
    
    train_AUC = []
    test_AUC = []
    holdout_with_attrs_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    confusion_matrix_rets = []
    calibrations = []
    
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    i = 0
    for train, test in cv.split(X, Y):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train], X.iloc[test]
        train_y, test_y = Y[train], Y[test]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy()
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex>=1':'sex'})
        
        ## remove unused feature in modeling
        train_x = train_x.drop(['person_id', 'screening_date', 'race'], axis=1)
        test_x = test_x.drop(['person_id', 'screening_date', 'race'], axis=1)
        cols=train_x.columns.tolist()
        
        ## add y to the data frame -- prepare for corel prediction
        train_data = train_x.copy()
        train_data['y'] = train_y
        train_data['index'] = train_data.index.tolist()
        
        test_data = test_x.copy()
        test_data['y'] = test_y
        test_data['index'] = test_data.index.tolist()
        
        ## build model
        COREL = corels.CorelsClassifier(n_iter=10000, 
                                        verbosity=[], 
                                        max_card=max_card, 
                                        c=c).fit(train_x, train_y, features=cols)
        
        ## extract features and signs
        Rule_features_sub, Rule_signs = extract_rules(model=COREL)
        train_prob, train_pred, train_auc = corel_prediction(dataset=train_data, 
                                                             rule_features_sub=Rule_features_sub,
                                                             rule_signs=Rule_signs)
        test_prob, test_pred, test_auc = corel_prediction(dataset=test_data, 
                                                          rule_features_sub=Rule_features_sub,
                                                          rule_signs=Rule_signs)
        
        ## fairness matrix
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
        
        train_AUC.append(train_auc)
        test_AUC.append(test_auc)
        holdout_with_attr_test.append(holdout_with_attrs)
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
    
    return {'train_auc':train_AUC, 
            'test_auc':test_AUC,
            'holdout_with_attrs_test': holdout_with_attr_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': df, 
            'calibration_stats': calibration_df}
    