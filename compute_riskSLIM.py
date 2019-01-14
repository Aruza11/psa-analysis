import sys,os
srcpath = "C:/Users/Caroline Wang/src/risk-slim-11-11-18"
sys.path.insert(0, srcpath)

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from utils import expand_grid

import cplex as cplex
from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa


# def predict(rho,dataX):
#     # score = np.apply_along_axis(np.dot(),0, dataX,a=rho)
#     score = np.dot(dataX, rho)
#     return score #a np array containing prediction  results

# score = predict(rho, test['X'])
# #TODO: REPLACE HARD CODED INTERCEPT WITH ACTUAL CODE
# intercept = model_info['solution'][0]
# test['prob_recid'] =  1/(1 + np.exp(intercept - score))


# print("The AUC of riskSLIM is: ",roc_auc_score(test['Y'], test['prob_recid']))
# print("Variabes are:\n")
# for i,name in enumerate(list(variable_names)): 
#     if rho[i]!=0: 
#         print(name, "\n")


def fit_riskSLIM(data, params): 
	'''
	@param data = {
		"X": train, (ndarray)
		"Y": test,  (ndarray)
		"sample_weights": sample_weights
		"variable_names": variable_names
	}
	@param params: dictionary containing parameters 
	'''

	settings = {
		# Problem Parameters
		'c0_value': params['c0_value'],
		'w_pos': params['w_pos'],
		#
		# LCPA Settings
		'max_runtime': 300.0,                               # max runtime for LCPA
		'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
		'display_cplex_progress': True,                     # print CPLEX progress on screen
		'loss_computation': 'normal',                       # how to compute the loss function ('normal','fast','lookup')
		#
		# RiskSLIM MIP settings
		'drop_variables': False,
		#
		# LCPA Improvements
		'round_flag': False,                                # round continuous solutions with SeqRd
		'polish_flag': False,                               # polish integer feasible solutions with DCD
		'chained_updates_flag': False,                      # use chained updates
		'initialization_flag': False,                       # use initialization procedure
		'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
		'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
		#
		# CPLEX Solver Parameters
		'cplex_randomseed': 0,                              # random seed
		'cplex_mipemphasis': 0,                             # cplex MIP strategy
	}

	# # turn on at your own risk
	# settings['round_flag'] = False
	# settings['polish_flag'] = False
	# settings['chained_updates_flag'] = False
	# settings['initialization_flag'] = False
	N, P = data["X"].shape

	# create coefficient set and set the value of the offset parameter
	coef_set = CoefficientSet(variable_names=data['variable_names'], lb=0, ub=params['min_coefficient'], sign=1)
	# print("MAX L0 VALUE IS", params['max_L0_value'])
	conservative_offset = get_conservative_offset(data, coef_set, int(params['max_L0_value']))
	max_offset = min(params['max_offset'], conservative_offset)
	coef_set['(Intercept)'].ub = max_offset
	coef_set['(Intercept)'].lb = -max_offset

	# create constraint
	trivial_L0_max = P - np.sum(coef_set.C_0j == 0)
	max_L0_value = min(int(params['max_L0_value']), trivial_L0_max)

	constraints = {
		'L0_min': 0,
		'L0_max': max_L0_value,
		'coef_set':coef_set,
	}

	# initialize MIP for lattice CPA
	mip_objects = setup_lattice_cpa(data, constraints, settings)

	# add operational constraints
	mip, indices = mip_objects['mip'], mip_objects['indices']
	get_alpha_name = lambda var_name: 'alpha_' + str(data['variable_names'].index(var_name))
	get_alpha_ind = lambda var_names: [get_alpha_name(v) for v in var_names]

	# to add a constraint like "either "CellSize" or "CellShape"
	# you must formulate the constraint in terms of the alpha variables
	# alpha[cell_size] + alpha[cell_shape] <= 1 to MIP
	# mip.linear_constraints.add(
	#         names = ["EitherOr_CellSize_or_CellShape"],
	#         lin_expr = [cplex.SparsePair(ind = get_alpha_ind(['UniformityOfCellSize', 'UniformityOfCellShape']),
	#                                      val = [1.0, 1.0])],
	#         senses = "L",
	#         rhs = [1.0])

	mip_objects['mip'] = mip

	# pass MIP back to lattice CPA so that it will solve
	model_info, mip_info, lcpa_info = finish_lattice_cpa(data, constraints, mip_objects, settings)

	#model info contains key results
	pprint(model_info)
	print_model(model_info['solution'], data)

	# mip_output contains information to access the MIP
	mip_info['risk_slim_mip'] #CPLEX mip
	mip_info['risk_slim_idx'] #indices of the relevant constraints

	# lcpa_output contains detailed information about LCPA
	pprint(lcpa_info)
	return model_info

#todo: implement cross validation on riskSLIM the way we do it in R
'''i.e. 1) Initialize an array of training parameters and split data into train, test 
		2) Loop through params, performing cross validation at each step to obtain auc summary statistics for model. 
		 Save in performance array 
			i) Cross validation procedure: 
			Shuffle the dataset randomly.
			Split the dataset into k groups
			For each unique group:
			Take the group as a hold out or test data set
			Take the remaining groups as a training data set
			Fit a model on the training set and evaluate it on the test set
			Retain the evaluation score and discard the model
			Summarize the skill of the model using the sample of model evaluation scores
		3) Pick model with highest mean c.v. auc. Train model on all data using these hyperparameters. 
		4) Evaluate on test set using auc. 

''' 
if __name__ == '__main__':

	# data
	train_name = "\\riskSLIM_train"                                      
	test_name = "\\riskSLIM_test"                                      # name of the data

	data_dir = os.getcwd()                                      # directory where datasets are stored
	train_csv = data_dir + train_name + '_data.csv'          # csv file for the train dataset
	test_csv = data_dir + test_name + '_data.csv'          # csv file for the test dataset

	# load data from disk
	data = load_data_from_csv(dataset_csv_file = train_csv)
	heldout_test_data = load_data_from_csv(dataset_csv_file = test_csv)

	# problem parameters
	params = {
		"max_coefficient" : [6],    #UNUSED?
		"min_coefficient" : [0],                                         # value of largest/smallest coefficient
		"max_L0_value" : [6],                                            # maximum model size
		"max_offset" : [50],                                             # maximum value of offset parameter (optional)
		"c0_value" : [1e-6],                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
		"w_pos" : [1.00]                                                 # relative weight on examples with y = +1; w_neg = 1.00 (optional)
	}

	setup = {
		"nfolds" : 5
	}


	param_grid = expand_grid(params) #list of dictionaries 
	nparam = len(param_grid)
	i_param_best = 0
	eval_varnames = dict.fromkeys(["train_auc_mean","train_auc_std","test_auc_mean","test_auc_std"])
	seeds = np.random.randint(10000, size = nparam)
	performance = [] #a list of dictionaries 

	for i, param_dict in enumerate(param_grid): 
		kfold = KFold(setup["nfolds"], shuffle = True, random_state = seeds[i])
		performance.append({
			"i_param": i, 
			"seed": seeds[i]}.update(eval_varnames)
			)

		for train_inds, test_inds in kfold.split(data['X'], data['Y']):
			train_folds = {
				"X": data['X'][train_inds],  #features
				"Y": data['Y'][train_inds],  #labels
				"sample_weights":  data['sample_weights'][train_inds],
				"variable_names":  data['variable_names']
			}

			fit_riskSLIM(train_folds , param_dict)
			# evaluate(data['X'][test_inds], data['Y'][test_inds])


	# #save model 
	# pickling_on = open("riskSLIM_model.pickle","wb")
	# pickle.dump(model_info, pickling_on)
	# pickling_on.close()

	# pickle_off = open("riskSLIM_model.pickle","rb")
	# model_info = pickle.load(pickle_off)

	# ###predict on held out test data ######
	# model_info = {'data_time': 0.429002046585083,
	#  'loss_value': 0.6242337422520895,
	#  'nodes_processed': 432081,
	#  'objective_value': 0.6242387422520895,
	#  'optimality_gap': 0.07066716970224067,
	#  'run_time': 300.0914874076843,
	#  'solution': np.array([-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,      0.,
	#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  0.,
	#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -0., -0., -0.,  4., -5.,
	#         0.,  1.,  0.,  0.,  0.,  0., -0.,  0.,  0.,  1.,  0., -0., -0.,
	#         0.,  0.]),
	#  'solver_time': 299.4157371520996,
	#  'w_pos': 1.0}

	# df = pd.read_csv(data_csv_file, sep=',')
	# raw_data = df.as_matrix()
	# data_headers = list(df.columns.values)
	# N = raw_data.shape[0]




