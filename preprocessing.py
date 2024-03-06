from typing import List
import numpy as np
from vendor import hamilton

def qrs_hamilton(data: np.array) -> List[int]:
	"""
	Preproceses ECG lead signals into list of QRS complexes.
	The final dimension (axis) of the data is assumed to be the samples for a single lead.

	LICENSE: GPL
	"""
	if len(data.shape) > 1:
		return [qrs_hamilton(data[i]) for i in range(data.shape[0])]
	else:
		return hamilton.detect_qrs(data)

def r_peaks_hamilton(data: np.array) -> List[int]:
	"""
	Preproceses ECG lead signals into list of R peaks.
	The final dimension (axis) of the data is assumed to be the samples for a single lead.

	LICENSE: GPL
	"""
	if len(data.shape) > 1:
		return [r_peaks_hamilton(data[i]) for i in range(data.shape[0])]
	else:
		return hamilton.detect_r_peaks(data)
