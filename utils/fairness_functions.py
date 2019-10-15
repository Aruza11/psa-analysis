import pandas as pd 
from sklearn.metrics import confusion_matrix

from utils.load_settings import load_settings

settings = load_settings()
decoders = settings["decoders"]


def compute_confusion_matrix_stats(df, preds, labels, protected_variables):
    df.loc[:, "score"] = preds
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score", 
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})
    
    rows = []
    for var in protected_variables:
        variable_summary = {}
        for value in df[var].unique():
            predictions = df["score"][df[var]==value]
            labels = df["label_value"][df[var]==value]
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()
            # predictive parity
            ppv = tp / (tp + fp)
            # false positive error rate balance
            fpr = fp / (fp + tn)
            # false negative error rate balance
            fnr = fn / (fn + tp)
            # equalized odds
            
            # conditional use accuracy equality
            
            # overall accuracy equality
            acc = (tp + tn)/ (tp + tn + fp + fn)
            
            # treatment equality
            ratio = fn / fp
            

            rows.append({
                "Attribute": var,
                "Attribute Value": value,
                "PPV": ppv,
                "FPR": fpr,
                "FNR": fnr,
                "Accuracy": acc,
                "Treatment Equality": ratio,
                "Individuals Evaluated On": len(labels)        
            })
    return pd.DataFrame(rows)


def compute_calibration_fairness(df, probs, labels, protected_variables):
    df.loc[:, "score"] = probs
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score", 
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})    
        
    rows = []
    for var in protected_variables:
        for value in df[var].unique():
            for window in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]:
                lo = window[0]
                hi = window[1]
                # get the individuals who have a predicted prob in the window (which we call their score)
                predictions = df["score"][(df[var]==value) & (df["score"] >= lo) & (df["score"] < hi)]
                labels = df["label_value"][(df[var]==value) & (df["score"] >= lo) & (df["score"] < hi)]
                
                # compute P(Y = 1 | Score = score, Protected_var = protected_var)
                prob = labels.sum() / len(labels)     

                rows.append({
                    "Attribute": var,
                    "Attribute Value": value,
                    "Lower Limit Score": lo,
                    "Upper Limit Score": hi,
                    "Conditional Frequency": prob,
                    "Individuals Evaluated On": len(labels)        
                })
    return pd.DataFrame(rows)


def compute_calibration_discrete_score(long_df:pd.DataFrame, 
                                        problem_name:str, 
                                        score_name:str) -> (pd.DataFrame, pd.DataFrame):
    """Returns dataframes of calibration values for discrete-valued score
    
    Keyword arguments: 
        long_df -- 
        problem_name -- 
        score_name -- 
    Returns:
        calib -- dataframe with the calibration values over all groups 
        calib_grps -- dataframe with the calibration values for each sensitive grp
    """
    # compute calibration overall
    calib = (long_df[[score_name, problem_name]]
                       .groupby(score_name)
                       .agg(['sum', 'size'])
                       .reset_index())

    calib.columns = [score_name, 'n_inds_recid', 'total_inds']
    calib["P(Y = 1 | Score = score)"] =  calib['n_inds_recid'] / calib['total_inds']
    
    # compute calibration for sensitive groups
    calib_grps = (long_df[[score_name, problem_name, 'Attribute Value']]
                           .groupby([score_name, 'Attribute Value'])
                           .agg(['sum', 'size'])
                           .reset_index())

    calib_grps.columns = [score_name, 'Attribute Value', 'n_inds_recid', 'total_inds']
    calib_grps["P(Y = 1 | Score = score, Attr = attr)"] =  calib_grps['n_inds_recid'] / calib_grps['total_inds']
    
    return calib, calib_grps


def parse_calibration_matrix(calibration_matrix: pd.DataFrame, 
                             problem_name:str, 
                             score_name:str):
    
    calibration_matrix[score_name] = (calibration_matrix["Lower Limit Score"].astype(str) + "-" 
                                     + calibration_matrix["Upper Limit Score"].astype(str))
    calibration_matrix.drop(columns = ['Lower Limit Score', 'Upper Limit Score'], inplace=True)
    calibration_matrix.rename(columns={'Conditional Frequency': "P(Y = 1 | Score = score, Attr = attr, Fold = fold)"}, 
                              inplace=True)
    
    # compute calibration by sensitive attribute (average out the fold)
    # filter entries where # inds evaluated on is 0
    calib_grps = (calibration_matrix[calibration_matrix["Individuals Evaluated On"] != 0]
                            .groupby([score_name, 'Attribute', 'Attribute Value']).apply(lambda x: np.average(x["P(Y = 1 | Score = score, Attr = attr, Fold = fold)"], 
                                                                         weights=x["Individuals Evaluated On"]))
                           .reset_index()
                           .rename(columns={0: "P(Y = 1 | Score = score, Attr = attr)"}))

    # put back the groups where the # inds eval on is 0
    num_inds = (calibration_matrix[[score_name, 'Attribute', 'Attribute Value', 'Individuals Evaluated On']]
                .groupby([score_name, 'Attribute', 'Attribute Value'])
                .sum()
                .reset_index())

    calib_grps = calib_grps.merge(num_inds, 
                                  on=[score_name, 'Attribute', 'Attribute Value'],
                                  how='right')

    # compute overall calib 
    calib = (calib_grps[calib_grps["Individuals Evaluated On"] != 0].groupby(score_name)
                       .apply(lambda x: np.average(x["P(Y = 1 | Score = score, Attr = attr)"], 
                                                   weights=x["Individuals Evaluated On"]))
                       .reset_index()
                       .rename(columns={0: 'P(Y = 1 | Score = score)'}))

    # num individuals w/ sensitive attrs per fold 
    num_inds = (calib_grps[[score_name, 'Individuals Evaluated On']]
                .groupby([score_name])
                .sum()
                .reset_index())

    calib = calib.merge(num_inds, 
                      on=[score_name],
                      how='right')

    return calib, calib_grps
