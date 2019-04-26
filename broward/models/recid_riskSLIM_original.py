import os
import numpy as np
import sys
basepath=os.path.dirname(__file__) #path of current script
root=os.path.abspath(os.path.join(basepath,".."))
sys.path.insert(0, root)

srcpath = "C:/Users/Caroline Wang/src/risk-slim-11-11-18"
sys.path.insert(0, srcpath)

from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
import cplex as cplex

##################################

train_name = "\\bin_train"                                      
data_dir = os.getcwd()                                      # directory where datasets are stored
train_csv = data_dir + train_name + '_data.csv'          # csv file for the train dataset

print(train_csv)

# load data from disk
subset = ["recid_use",
  "p_current_age18", "p_current_age1929", "p_current_age3039", "p_current_age4049", "p_current_age5059", "p_current_age60plus",
  "p_property0","p_property13", "p_property46", "p_property79", "p_property10up", 
  "prior_conviction_M01", "prior_conviction_M24", "prior_conviction_M57", "prior_conviction_M810" , "prior_conviction_M11up",
  "p_charge0", "p_charge13", "p_charge45", "p_charge67", "p_charge810" , "p_charge11up", 
  "p_felprop_violarrest0", "p_felprop_violarrest1", "p_felprop_violarrest2", "p_felprop_violarrest3", "p_felprop_violarrest46", "p_felprop_violarrest7up",  
  "total_convictions0", "total_convictions1", "total_convictions2", "total_convictions3", "total_convictions46", "total_convictions7up"]

data = load_data_from_csv(dataset_csv_file = train_csv, subset = subset, sample_weights_csv_file = None)

# 
# problem parameters
max_coefficient = 5                                         # value of largest/smallest coefficient
max_L0_value = 5                                            # maximum model size (set as float(inf))
max_offset = 50                                             # maximum value of offset parameter (optional)
c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
w_pos = 1.00                                                # relative weight on examples with y = +1; w_neg = 1.00 (optional)

# load data from disk

# create coefficient set and set the value of the offset parameter
coef_set = CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
max_offset = min(max_offset, conservative_offset)
coef_set['(Intercept)'].ub = max_offset
coef_set['(Intercept)'].lb = -max_offset

constraints = {
    'L0_min': 0,
    'L0_max': max_L0_value,
    'coef_set':coef_set,
}


# major settings (see riskslim_ex_02_complete for full set of options)
settings = {
    # Problem Parameters
    'c0_value': c0_value,
    'w_pos': w_pos,
    #
    # LCPA Settings
    'max_runtime': 30.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
    #
    # LCPA Improvements
    'round_flag': True,                                # round continuous solutions with SeqRd
    'polish_flag': True,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': True,                      # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # Initialization
    'initialization_flag': True,                       # use initialization procedure
    'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
    'init_max_coefficient_gap': 0.49,
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}

# train model using lattice_cpa
model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)

#model info contains key results
pprint(model_info)
print_model(model_info['solution'], data)

# mip_output contains information to access the MIP
mip_info['risk_slim_mip'] #CPLEX mip
mip_info['risk_slim_idx'] #indices of the relevant constraints

# lcpa_output contains detailed information about LCPA
pprint(lcpa_info)
