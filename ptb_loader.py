# Partially sourced from physionet ptb-xl

import pandas as pd
import numpy as np
import wfdb
import ast
import plotly.express as px
import code
import pickle
import os
from tqdm import tqdm
from enum import Enum

CACHE_DIR = "./.cache/"
CACHE_VER = 6

class PTBDataset:
	__TEST_FOLD = 10

	def __init__(self, x_digital: np.array, x_analog: np.array, y: pd.DataFrame):
		self.__x_digital = x_digital
		self.__x_analog = x_analog
		self.__y = y

	def __type_cond(self, type: str) -> np.array:
		if type == "train":
			return self.__y.strat_fold != PTBDataset.__TEST_FOLD
		elif type == "test":
			return self.__y.strat_fold == PTBDataset.__TEST_FOLD
		else:
			raise ValueError(f"Unexpected type {type}")

	def x(self, type: str, digital: bool):
		data = self.__x_digital if digital else self.__x_analog
		return data[np.where(self.__type_cond(type))]

	def y(self, type: str):
		return self.__y[self.__type_cond(type)]

def record_to_signals(r):
	return (r.d_signal.astype(np.int16), r.dac())


def load_raw_data(df: pd.DataFrame, dataset: str, path: str):
	filenames = df[f"filename_{dataset}"]
	digital = []
	analog = []
	for f in tqdm(filenames, desc="Loading data"):
		record = wfdb.rdrecord(path+f, return_res=16, physical=False)
		digital.append(record.d_signal)
		analog.append(record.dac())

	return (
		np.array(digital).transpose((0,2,1)), 
		np.array(analog).transpose((0,2,1))
	)

def aggregate_diagnostic(agg_df, y_dic):
	tmp = []
	for key in y_dic.keys():
		if key in agg_df.index:
			tmp.append(agg_df.loc[key].diagnostic_class)
	return list(set(tmp))


def load_ptb_xl(path: str, dataset: str) -> PTBDataset:
	cache_path = os.path.join(CACHE_DIR, f"ptb-xl-v{CACHE_VER}-{dataset}.pkl")
	try:
		os.makedirs(CACHE_DIR, exist_ok=True)
		with open(cache_path, "rb") as f:
			print("Loading PTB-XL from cache...")
			obj = pickle.loads(f.read())
			print("PTB-XL loaded from cache!")
			return obj
	except FileNotFoundError:
		print("Cached PTB-XL not found.")
		pass
		

	# load and convert annotation data
	print("Loading csv...")
	Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
	print("Converting annotation data...")
	Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

	# Load raw signal data
	(X_digital, X_analog) = load_raw_data(Y, dataset, path)

	# Load scp_statements.csv for diagnostic aggregation
	print("Applying diagnostic data...")
	agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
	agg_df = agg_df[agg_df.diagnostic == 1]

	# Apply diagnostic superclass
	Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(agg_df, x))

	dataset = PTBDataset(X_digital, X_analog, Y)

	print("Caching loaded PTB-XL...")
	with open(cache_path, "wb") as f:
		f.write(pickle.dumps(dataset))
	print("PTB-XL loaded!")

	return dataset
	
