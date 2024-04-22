from typing import List, Optional, Tuple
from preprocessing import qrs_hamilton
from dataclasses import dataclass, fields
from ptb_loader import PTBDataset, load_ptb_xl
from itertools import combinations
import time
import threading
import numpy as np
import pandas as pd
from util import Spinner, cached
from tqdm import tqdm
import argparse
from os import path
from autogluon.tabular import TabularDataset, TabularPredictor

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
			rr_average_x=float("NaN"),
			rr_variance_x=float("NaN"),
			qr_average_y=float("NaN"),
			qr_variance_y=float("NaN"),
			rs_average_y=float("NaN"),
			rs_variance_y=float("NaN"),
			qs_average_x=float("NaN"),
			qs_variance_x=float("NaN"),
			gr_average_y=float("NaN"),
			gr_variance_y=float("NaN"),
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

def tabulate(y_df: pd.DataFrame, x_lead_features: List[List[LeadFeatures]], leads: Tuple[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	names = [f.name for f in fields(LeadFeatures)]
	data = {}
	for lead in range(leads[0], leads[1]):
		for name in names:
			data[f"lead_{lead}_{name}"] = np.zeros(len(x_lead_features))
	
	for i, patient in enumerate(x_lead_features):
		for j, lead in enumerate(patient, leads[0]):
			for name in names:
				data[f"lead_{j}_{name}"][i] = getattr(lead, name)
	
	x = pd.DataFrame(data=data)
	y = pd.DataFrame(data={
		"is_normal": map(lambda x: "NORM" in x, y_df["diagnostic_superclass"]),
	})
	
	return x, y


def automl_model(train_data: pd.DataFrame, file: str) -> TabularPredictor:
	if file is not None and path.exists(file):
		print(f"AutoGluon predictor {file} already exists - loading from file")
		return TabularPredictor.load(file)
	else:
		print(f"Training AutoGluon predictor {file}...")
		return TabularPredictor(label="is_normal").fit(
			train_data,
			presets = "best_quality",
			ag_args_fit={
				"num_cpus": 16,
				"num_gpus": 0
			}
		)

def get_data(ptb: PTBDataset, data_type: str, leads: Tuple[int, int]):
	def extract():
		with Spinner(f"QRS Detection {data_type}"):
			qrs = qrs_hamilton(ptb.x(data_type, True)[:,leads[0]:leads[1],:])
		return __extract_features(ptb.x(data_type, False)[:,leads[0]:leads[1]], qrs)
	
	qrs_features = cached(f"data_{data_type}_{leads}", 1, extract)

	x, y = tabulate(ptb.y(data_type), qrs_features, leads)
	return pd.concat([x, y], axis=1)

def run(file: str, leads: Tuple[int, int]):
	ptb = load_ptb_xl("./ptb-xl/", "lr")
	
	train_data = get_data(ptb, "train", leads)
	test_data = get_data(ptb, "test", leads)

	predictor = automl_model(train_data, file)

	print("=== LEADERBOARD ===")
	print(predictor.leaderboard(test_data))
	print("=== EVALUATE ===")
	print(predictor.evaluate(test_data, detailed_report=True))

def parse_leads(l: str) -> Tuple[int, int]:
	split = l.split(":")
	tup = (-1, -1)
	if len(split) == 1:
		tup = (0, int(split[0]))
	elif len(split) == 2:
		tup = (int(split[0]) - 1, int(split[1]))
	if tup[0] < 0 or tup[1] < 0 or tup[0] >= 12 or tup[1] >= 12:
		print("Leads must be in range 0:0 to 12:12")
		exit(1)
	return tup
	

def main():
	parser = argparse.ArgumentParser(prog="Heartly AutoML")
	parser.add_argument("-f", "--file")
	parser.add_argument("-l", "--leads", default="12", type=str)
	args = parser.parse_args()
	run(file=args.file, leads=parse_leads(args.leads))

if __name__ == "__main__":
	main()
