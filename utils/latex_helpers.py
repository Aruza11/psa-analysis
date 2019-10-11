import pytablewriter
import pandas as pd 


def df_to_latex(df: pd.DataFrame):
	values = df.values.tolist()

	writer = pytablewriter.LatexTableWriter()
	writer.value_matrix = values
	writer.write_table()

	return

