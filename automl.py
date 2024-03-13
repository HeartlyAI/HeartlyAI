from typing import List, Optional, Tuple
from preprocessing import qrs_hamilton
from dataclasses import dataclass
from ptb_loader import PTBDataset, load_ptb_xl
import time
import threading
import numpy as np
import pandas as pd
from util import Spinner, cached
from tqdm import tqdm
from autogluon.tabular import TabularDataset, TabularPredictor

label = 'is_norm'
presets = 'best_quality'
time_limit = 60*30
auto_stack = True
dynamic_stacking = False
num_stack_levels = None
num_bag_folds = None
num_bag_sets = None
excluded_model_types = ["KNN"]

QRSList = List[List[List[List[int]]]]

# Steps:
# - Preprocess huge dataset
# - Train automl model on resulting dataset

# Future goal: Run preproccessor and model in realtime.

@dataclass
class LeadFeatures:
	rr_average_x: float
	rr_variance_x: float
	qr_average_y: float
	qr_variance_y: float
	rs_average_y: float
	rs_variance_y: float
	qs_average_x: float
	qs_variance_x: float
	# Average between G (center of QRS) and R
	gr_average_y: float
	# Delta between G (center of QRS) and R
	gr_variance_y: float
		

def __find_triangle_center(ax, ay, bx, by, cx, cy):
    # Calculate midpoints of sides
	m1x, m1y = (ax + bx) / 2, (ay + by) / 2
	m2x, m2y = (bx + cx) / 2, (by + cy) / 2
	m3x, m3y = (ax + cx) / 2, (ay + cy) / 2

	# Find centroid coordinates
	c = np.zeros((2, *ax.shape))
	c[0] = (m1x + m2x + m3x) / 3
	c[1] = (m1y + m2y + m3y) / 3
	return c

def __extract_features(data: np.array, qrs: QRSList) -> List[List[LeadFeatures]]:
	return [__extract_patient_features(*v) for v in zip(data, tqdm(qrs))]

def __extract_patient_features(data: np.array, patient: List[List[List[int]]]) -> List[LeadFeatures]:
	return [__extract_lead_features(*v) for v in zip(data, patient)]

def __extract_lead_features(data: np.array, lead: List[List[int]]) -> LeadFeatures:
	def clip_qrs(q, r, s):
		if len(q) > 0 and q[0] == -1: 
			# if the first elem of q is -1 it means we are missing a q extrema for the first r peak. 
			# Action is then to disregard this peak
			q = q[1:]
			r = r[1:]
			s = s[1:]

		if len(s) > 0 and s[-1] == -1:
			# if the last elem of s is -1 it means we are missing a s extrema for the last r peak. 
			# Action is then to disregard this peak
			q = q[:-2]
			r = r[:-2]
			s = s[:-2]
		return q, r, s

	q, r, s = map(np.array, lead)
	cq, cr, cs = clip_qrs(q, r, s)

	if (len(q) == 0):
		return LeadFeatures(
			rr_average_x=float('NaN'),
			rr_variance_x=float('NaN'),
			qr_average_y=float('NaN'),
			qr_variance_y=float('NaN'),
			rs_average_y=float('NaN'),
			rs_variance_y=float('NaN'),
			qs_average_x=float('NaN'),
			qs_variance_x=float('NaN'),
			gr_average_y=float('NaN'),
			gr_variance_y=float('NaN'),
		)
	
	# Calculate features for the current lead
	qrs_center = __find_triangle_center(cq, data[cq], cr, data[cr], cs, data[cs])

	# R-R intervals
	rr_delta_x = np.diff(r)
	qs_delta_x = cs - cq
	qr_delta_y = data[cr] - data[cq]
	rs_delta_y = data[cr] - data[cs]
	gr_delta_y = data[cr] - qrs_center[1]

	return LeadFeatures(
		rr_average_x=np.average(rr_delta_x), 
		rr_variance_x=np.var(rr_delta_x),
		qr_average_y=np.average(qr_delta_y),
		qr_variance_y=np.var(qr_delta_y),
		rs_average_y=np.average(rs_delta_y),
		rs_variance_y=np.var(rs_delta_y),
		qs_average_x=np.average(qs_delta_x),
		qs_variance_x=np.var(qs_delta_x),
		gr_average_y=np.average(gr_delta_y),
		gr_variance_y=np.var(gr_delta_y),
	)

def tabulate(ptb: PTBDataset, x_lead_features: List[List[LeadFeatures]]):
	X_train = ptb.x("train", False)
	Y_train = ptb.y("train")
	#print(set(Y_train["diagnostic_superclass"].to_numpy()))
	df = pd.DataFrame(data={
		"normal": map(lambda x: "NORM" in x, Y_train["diagnostic_superclass"]),
	})
	print(df)
	exit(0)
	df = Y_train.copy()
	print(df.iloc[0])
	exit(0)

	# Add result columns to the expected output dataset


def train_model(train_data):
	predictor = TabularPredictor(label=label).fit(
		train_data,
		time_limit=time_limit, 
		presets=presets, 
		auto_stack=auto_stack, 
		num_stack_levels=num_stack_levels,
		num_bag_folds=num_bag_folds, 
		num_bag_sets=num_bag_sets, 
		excluded_model_types=excluded_model_types, 
		dynamic_stacking=dynamic_stacking
	)

	return predictor


def main():
	ptb = load_ptb_xl("./ptb-xl/", "lr")
	with Spinner("QRS Detection"):
		qrs = cached("hamilton", 2, lambda: qrs_hamilton(ptb.x("train", True)))

	qrs_features = cached("feature_extract", 1, lambda: __extract_features(ptb.x("train", False), qrs))
	
	train_data = tabulate(ptb, qrs_features)
	predictor = train_model(train_data)
	# predictor.evaluate(test_data)
	print("Gabagooye")
	pass

if __name__ == "__main__":
	main()
