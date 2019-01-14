import pandas as pd
from itertools import product


def expand_grid(param_dict):
	'''
	@params: dictionary where each key is a possible parameter and 
	each item is a list of possible parameter values 
	returns: a list of dicts where each dict holds unique combination 
			of possible parameter values
	'''
	param_df = pd.DataFrame([row for row in product(*param_dict.values())], 
					   columns=param_dict.keys())

	param_grid = []
	for i, row in param_df.iterrows(): 
		param_grid.append({colname:row[colname] for colname in list(param_df)})
	return param_grid
