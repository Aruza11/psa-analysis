import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting_helpers import safe_save_plt


def plot_calibration_for_score_on_problem(calib: pd.DataFrame, 
                                          calib_grps: pd.DataFrame,
                                          problem_name: str,
                                          score_name: str,
                                          region: str,
                                          xtick_labels=None,
                                          rotate_xticks=False,
                                          save_path=None):
    """Plots calibration for risk scores which are not probabilities
    Keyword Arguments:
        calib: df with columns [score_name, P(Y = 1 | Score = score)]; should contain the probability of 
                recidivism for each score. 
        calib_grps: df with columns [score_name, Attribute Value, P(Y = 1 | Score = score, Attr=attr)]; 
                                should contain the prob of recidivism for each sensitive group and score 
        problem_name: variable name of the prediction problem for this plot 
        score_name: 
    """
    
    plt.figure(figsize=(10, 7))
    plt.style.use('ggplot')

    # calibration reference line doesn't make sense here
    # overall calibration 
    plt.plot(calib[score_name], 
             calib['P(Y = 1 | Score = score)'], 
             color='black', marker='o', 
             linewidth=1, markersize=4,
             label='All individuals')

    # group level calibration 
    colors=['red', 'green', 'orange', 'maroon', 'royalblue', 'mediumslateblue']
    for i, (name, group_df) in enumerate(calib_grps.groupby("Attribute Value")):
        if name == 'female' or name == 'male':
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=4,
                     linestyle='--',
                     label=name)
        else:
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=4,
                     label=name)

    # axes settings
    if xtick_labels is not None:
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels)
    plt.xlabel(f"\n{score_name} score", fontsize=25)

    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Score = score, Attr = attr)\n', fontsize=25)

    # Create legend, add title, format & show/save graphic
    plt.legend(fontsize=20)
    score_name_formatted = score_name.capitalize()
    plt.title(f'Calibration of {score_name_formatted} on \n{problem_name} in {region}', fontsize=30)

    if rotate_xticks:
        plt.tick_params(axis="x", labelsize=20, rotation=45)
    else:
        plt.tick_params(axis="x", labelsize=20)

    plt.tick_params(axis="y", labelsize=20)

    if save_path is not None: 
        safe_save_plt(plt, save_path)
        
    plt.show()
    plt.close()
    return


def prob_recid_conditioned_sensitive_attr(df:pd.DataFrame, 
                                          attribute:str, 
                                          dataset_name:str,
                                          save_path:None):
    # cast df from long to wide with each attribute being a different column 
    wide_df = (df[df["Attribute"] == attribute]
                .pivot(index='label', 
                       columns='Attribute Value', 
                       values=[ 'P(Y = 1 | Attr = attr)']))
    
    # get a list of unique columns
    attribute_values = df[df["Attribute"] == attribute ]["Attribute Value"].unique()
    
    # set width of bar
    barWidth = 0.15

    # set height of bar
    bars = {attribute_value: {"bar": None, "pos": None} for attribute_value in attribute_values}
    for attribute_value in attribute_values:
        bars[attribute_value]["bar"] = wide_df[('P(Y = 1 | Attr = attr)', attribute_value)]
        bar_len = len(bars[attribute_value]["bar"])

    # Set position of bar on X axis
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        if i == 0:
            bar_dict["pos"] = np.arange(len(bar_dict["bar"]))
        else: 
            prev_bar_pos = bars[prev_bar_name]["pos"]
            bar_dict["pos"] = [x + barWidth for x in prev_bar_pos]
        prev_bar_name = bar_name 

    # Make the plot
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    
    colors = ['cornflowerblue', 'lightslategrey', 'lightskyblue', 'steelblue']
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        plt.bar(bar_dict["pos"], bar_dict["bar"], color=colors[i], width=barWidth, edgecolor='white', label=bar_name)

    # Add xticks on the middle of the group bars
    plt.xlabel('Prediction Problem', fontweight='bold')
    plt.xticks([r + barWidth for r in range(bar_len)], wide_df.index, rotation=45)

    # Limit y axis to 0,1 
    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Attr = attr)', fontweight='bold')

    # Create legend, add title, & show/save graphic
    plt.legend()
    plt.title(f'Probability of recidivism (conditioned on {attribute}) is not the same for \nany prediction problem on {dataset_name}')
    
    if save_path is not None: 
        safe_save_plt(plt, save_path)
    plt.show()
    plt.close()
    return
    