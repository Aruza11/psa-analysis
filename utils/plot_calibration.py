import pandas as pd 
import matplotlib.pyplot as plt

from utils.plotting_helpers import safe_save_plt
from utils.fairness_functions import compute_calibration_discrete_score


def plot_calibration_for_score_on_problem(long_df: pd.DataFrame, 
                                           problem_name: str,
                                           score_name: str,
                                           region: str,
                                           save_path=None):
    """Plots calibration for risk scores which are not probabilities
    Keyword Arguments:
        scores: df with columns [score_name, P(Y = 1 | Score = score)]; should contain the probability of 
                recidivism for each score. 
        sensitive_attr_scores: df with columns [score_name, Attribute Value, P(Y = 1 | Score = score, Attr=attr)]; 
                                should contain the prob of recidivism for each sensitive group and score 
        problem_name: variable name of the prediction problem for this plot 
        score_name: 
    """
    calib, calib_grps = compute_calibration_discrete_score(long_df=long_df, 
                                                            problem_name=problem_name, 
                                                            score_name=score_name)
    
    plt.figure(figsize=(10, 7))
    plt.style.use('ggplot')

    # calibration reference line doesn't make sense here
    # overall calibration 
    plt.plot(calib[score_name], 
             calib['P(Y = 1 | Score = score)'], 
             color='black', marker='o', 
             linewidth=1, markersize=2)

    # group level calibration 
    colors=['red', 'green', 'orange', 'maroon', 'royalblue', 'mediumslateblue']
    for i, (name, group_df) in enumerate(calib_grps.groupby("Attribute Value")):
        if name == 'female' or name == 'male':
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=2,
                     linestyle='--',
                     label=name)
        else:
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=2,
                     label=name)

    # axes settings
    plt.xlabel(f"\n{score_name} Score", fontsize=20)

    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Score = score, Attr = attr)\n', fontsize=20)

    # Create legend, add title, & show/save graphic
    plt.legend(fontsize=14)
    plt.title(f'Calibration of {score_name} on \n{problem_name} in {region}', fontsize=24)

    if save_path is not None: 
        safe_save_plt(plt, save_path)
        
    plt.show()
    plt.close()
    return
    