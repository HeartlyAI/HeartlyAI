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
from util import Spinner, Cache

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

	def x(self, type: str, digital: bool) -> np.array:
		data = self.__x_digital if digital else self.__x_analog
		return data[np.where(self.__type_cond(type))]

	def y(self, type: str) -> pd.DataFrame:
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
	cache = Cache[int](f"ptb-xl-{dataset}", CACHE_VER)
	with Spinner("Loading PTB-XL from cache"):
		obj = cache.try_load()
	if obj is not None:
		return obj
	else:
		print("Cached PTB-XL not found")

	# load and convert annotation data
	with Spinner("Loading csv"):
		Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
	with Spinner("Converting annotation data"):
		Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

	# Load raw signal data
	(X_digital, X_analog) = load_raw_data(Y, dataset, path)

	# Load scp_statements.csv for diagnostic aggregation
	with Spinner("Applying diagnostic data"):
		agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
		agg_df = agg_df[agg_df.diagnostic == 1]

	# Apply diagnostic superclass
	Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(agg_df, x))

	dataset = PTBDataset(X_digital, X_analog, Y)

	with Spinner("Caching loaded PTB-XL"):
		cache.save(dataset)

	return dataset
	
