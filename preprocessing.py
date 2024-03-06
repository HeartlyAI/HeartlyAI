from typing import List
import numpy as np
from vendor import hamilton



def r_peaks_hamilton(data: np.array) -> List[int]:
	"""
	Preproceses a single ECG lead signal into a list of R peaks.
	The final dimension (axis) of the data is assumed to be the samples for a single lead.

	LICENSE: GPL
	"""
	try:
		from vendor import hamilton
		if len(data.shape) > 1:
			return [r_peaks_hamilton(data[i]) for i in range(data.shape[0])]
		else:
			return hamilton.detect_r_peaks(data)
	except ImportError:
		pass
