from ptb_loader import load_ptb_xl
import plotly.express as px
import scipy
import numpy as np
import code

dataset = load_ptb_xl("./ptb-xl/", "lr")
X_train = dataset["train"][0]
y_train = dataset["train"][1]

def ecg_bandpass(data, sampling_freq):
	def butter_bandpass(lowcut, highcut, fs, order):
		nyquist = fs / 2
		low = lowcut / nyquist
		high = highcut / nyquist
		b, a = scipy.signal.butter(order, [low, high], btype='band')
		return b, a


	def butter_bandpass_filter(data, lowcut, highcut, fs, order):
		b, a = butter_bandpass(lowcut, highcut, fs, order=order)
		y = scipy.signal.lfilter(b, a, data)
		return y

	return butter_bandpass_filter(data, 3, 45, sampling_freq, 5)

def ecg_hamilton(data, sampling_freq):
	"""
	Based on http://www.lx.it.pt/~afred/papers/Review%20and%20Comparison%20of%20Real%20Time%20Electrocardiogram%20Segmentation%20Algorithms%20for%20Biometric%20Applications.pdf
	"""
	# 1. ignore all peaks that precede or follow larger peaks by less than 200ms;
	# 2. if the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise;
	# 3. if an interval equal to 1.5 times the average R-to-R interval has 
	#    elapsed since the most recent detection, within that interval there 
	#    was a peak that was larger than half the detection threshold, and the
	#    peak followed the preceding detection by at least 360ms, classify that
	#    peak as a QRS complex;
	# 4. the detection threshold is a function of the average noise and the
	#    average QRS peak values;
	# 5. the average noise peak, the average QRS peak and the average
	#    R-to-R interval estimates are calculated as the mean/media
	#    of the last eight values.

	# 8-Point Median Peak for Estimation
	# Detection threshold coeff: 0. 1825
	# In our case, a peak is only considered when it is the higher value in a window interval, this way multiple detection is avoided.

	# Preprocessing Stage
	# Band-pass filtering
	data = ecg_bandpass(data, sampling_freq)
	# Derivative
	data = scipy.signal.savgol_filter(data, 5, 1)
	# Squaring
	data = data**2

	# Detection Stage
	window_size = int(0.15 * sampling_freq)
	n_signals = data.shape[1]
	window = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)
	np.unique(np.argmax(window, axis=3), axis=1)
	code.interact(local=locals())
	

def ecg_pan_tompkins(data, sampling_freq):

	pass

ecg_hamilton(X_train[0], 100)

#px.line(ecg_bandpass(X_train[0], 100), title="TETSSET").show()
#px.line(X_train[0], title="TETSSET").show()
