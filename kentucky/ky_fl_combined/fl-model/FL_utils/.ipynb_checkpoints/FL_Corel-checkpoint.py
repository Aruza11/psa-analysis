from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd 
import numpy as np
import corels

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
    proba = (prediction_table['probability'] > 0.5).values
    pred = prediction_table['probability'].values
    auc = roc_auc_score(prediction_table['y'], proba)
    
    return proba, pred, auc



def corel_cv(KY_x, KY_y, FL_x, FL_y, max_card, c, seed):
    
    cols=KY_x.columns.tolist()
    KY_y[KY_y == -1] = 0
    FL_y[FL_y == -1] = 0
    
    KY_data = KY_x.copy()
    KY_data['y'] = KY_y
    KY_data['index'] = KY_data.index.tolist()
    
    FL_data = FL_x.copy()
    FL_data['y'] = FL_y
    FL_data['index'] = FL_data.index.tolist()
    
    COREL = corels.CorelsClassifier(n_iter=10000, verbosity=[], 
                                    max_card=max_card, c=c).fit(FL_x, FL_y, features=cols)
    
    ## extract features and signs
    Rule_features_sub, Rule_signs = extract_rules(model=COREL)
    KY_prob, KY_pred, KY_score = corel_prediction(dataset=KY_data, 
                                                  rule_features_sub=Rule_features_sub,
                                                  rule_signs=Rule_signs) 
    FL_prob, FL_pred, FL_score = corel_prediction(dataset=FL_data, 
                                                  rule_features_sub=Rule_features_sub,
                                                  rule_signs=Rule_signs)    

        
    return KY_score, FL_score
    