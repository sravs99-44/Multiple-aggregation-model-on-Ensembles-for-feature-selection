import pandas as pd
import numpy as np
import fs

def df_arrange(dataset,ranks):
  colms = []
  df = pd.read_csv(dataset)
  col = df.columns
  for i in ranks:
    colms.append(col[i-1])
  colms.append(col[-1])
  df_new = pd.DataFrame(columns = colms)
  for i in colms:
    df_new[i] = df[i]
  return df_new

def filter(filter,dataset):
	df = pd.read_csv(dataset)
	data = fs.bootstrap(df)
	if filter == "reliefF":
		R = fs.relieF(data)
		new_data = df_arrange(dataset,R)
		return fs.dispcolumns(df,R),new_data

	elif filter == "fisher":
		R = fs.fisher(data)
		new_data = df_arrange(dataset,R)
		return fs.dispcolumns(df,R),new_data
	elif filter == "mim":
		R = fs.mim(data)
		new_data = df_arrange(dataset,R)
		return (fs.dispcolumns(df,R)),new_data
	elif filter == "icap":
		R = fs.icap(data)
		new_data = df_arrange(dataset,R)
		return (fs.dispcolumns(df,R)),new_data
	elif filter == "cmim":
		R = fs.cmim(data)
		new_data = df_arrange(dataset,R)
		return (fs.dispcolumns(df,R)),new_data
	elif filter == "jmi":
		R = fs.jmi(data)
		new_data = df_arrange(dataset,R)
		return (fs.dispcolumns(df,R)),new_data
	elif filter == "disr":
		R = fs.disr(data)
		new_data = df_arrange(dataset,R)
		return (fs.dispcolumns(df,R)),new_data
	else:
		return ("select the filter first")
