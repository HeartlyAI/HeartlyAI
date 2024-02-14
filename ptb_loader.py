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
CACHE_VER = 1

def load_raw_data(df: pd.DataFrame, dataset: str, path: str):
	filenames = df[f"filename_{dataset}"]
	data = [wfdb.rdsamp(path+f) for f in tqdm(filenames, desc="Loading data")]
	data = np.array([signal for signal, meta in data])
	return data

def aggregate_diagnostic(agg_df, y_dic):
	tmp = []
	for key in y_dic.keys():
		if key in agg_df.index:
			tmp.append(agg_df.loc[key].diagnostic_class)
	return list(set(tmp))


def load_ptb_xl(path: str, dataset: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
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
	X = load_raw_data(Y, dataset, path)

	# Load scp_statements.csv for diagnostic aggregation
	print("Applying diagnostic data...")
	agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
	agg_df = agg_df[agg_df.diagnostic == 1]

	# Apply diagnostic superclass
	Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(agg_df, x))

	# Split data into train and test
	test_fold = 10
	# Train
	X_train = X[np.where(Y.strat_fold != test_fold)]
	y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
	# Test
	X_test = X[np.where(Y.strat_fold == test_fold)]
	y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

	obj = {
		"train": (X_train, y_train),
		"test": (X_test, y_test)
	}

	print("Caching loaded PTB-XL...")
	with open(cache_path, "wb") as f:
		f.write(pickle.dumps(obj))
	print("PTB-XL loaded!")

	return obj
	
