import pandas as pd 

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


def compute_fairness(df: pd.DataFrame,
                     decoders: dict,
                     sensitive_attrs: list,
                     ref_groups_dict: dict) -> pd.DataFrame:
    """
    decoders: dictionary of dictionary of decoders 
    """
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
    return fdf[['attribute_name', 'attribute_value'] + absolute_metrics + b.list_disparities(fdf) + parity_determinations].style
