import pandas as pd 

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
from sklearn.metrics import confusion_matrix

# some hard-coded attrs

decoders = {"sex": {0: "male",
                    1: "female"}, 
            "race": {"White": "Caucasian",
                     "Black": "African-American",
                     "Race Unknown": "Other"} # indian or native american?
            }

sensitive_attrs = ['sex', 'race']

ref_groups_dict = {'sex': 'male',
                   'race': 'Caucasian'}

def compute_fairness(df: pd.DataFrame,
                     preds, 
                     labels) -> pd.DataFrame:
    """
    decoders: dictionary of dictionary of decoders 
    """
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

    g = Group()
    xtab, _ = g.get_crosstabs(df, attr_cols=sensitive_attrs)
    # compute bias
    b = Bias()
    bdf = b.get_disparity_predefined_groups(xtab,
                                            original_df=df,
                                            ref_groups_dict=ref_groups_dict,
                                            alpha=0.05,
                                            # check_significance=True,
                                            # mask_significance=True
                                            )
    f = Fairness()
    fdf = f.get_group_value_fairness(bdf)

    # list results of fairness analysis
    parity_determinations = f.list_parities(fdf)

    absolute_metrics = g.list_absolute_metrics(xtab)
    return fdf[['attribute_name', 'attribute_value'] + absolute_metrics + b.list_disparities(fdf) + parity_determinations]

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
    
    confusion_matrix_fairness = {}
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
            
            output_summary = {
                "PPV": ppv,
                "FPR": fpr,
                "FNR": fnr,
                "Accuracy": acc,
                "Treatment Equality": ratio
            }
            variable_summary[value] = output_summary
        confusion_matrix_fairness[var] = variable_summary
    return confusion_matrix_fairness
    