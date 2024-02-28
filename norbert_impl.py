from ptb_loader import load_ptb_xl
import plotly.graph_objects as go
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

	return butter_bandpass_filter(data, 8, 16, sampling_freq, 5)

def ecg_pass(data, cutoff, sampling_freq, btype):
	nyquist = sampling_freq / 2
	cut = cutoff / nyquist
	b, a = scipy.signal.butter(5, cut, btype=btype)
	y = scipy.signal.lfilter(b, a, data)
	return y

def ecg_low_pass(data, lowcut, sampling_freq):
	y = ecg_pass(data, lowcut, sampling_freq, 'lowpass')
	return y

def ecg_high_pass(data, highcut, sampling_freq):
	y = ecg_pass(data, highcut, sampling_freq, 'highpass')
	return y

def detect_peaks(data, plateu_size=3, min_time_between=30):
	data_peaks = []
	n_signals = data.shape[1]
	for i in range(n_signals):
		peaks = []
		signal = data[:, i]
		prev_value = 1.0
		plateu = 0
		prev_direction = 0
		prev_plateu_index = -999999
		threshold = np.max(signal) // 2
		for j in range(len(signal)):
			value = signal[j]
			direction = value-prev_value
			if (plateu != 0 or (plateu == 0 and prev_direction >= 0)) and value > threshold and direction >= -1e-9 and direction <= 1e-9:
				plateu += 1
			else:
				plateu = 0
			if plateu >= plateu_size and (j - prev_plateu_index) > min_time_between:
				peaks.append(j)
				plateu = 0
				prev_plateu_index = j
				threshold = prev_value // 2
			prev_value = value
			prev_direction = direction
		data_peaks.append(peaks)
	return data_peaks

def ecg_hamilton(data, sampling_freq):
	"""
	Based on https://ieeexplore.ieee.org/document/1166717
	"""
	# The Hamilton QRS pre-processing is as follows:
	# 1. Band-pass filtering 8Hz - 16Hz
	# 2. Derivative
	# 3. Moving average w/ 80ms window

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

	# Preprocessing Stage
	# Band-pass filtering
	filtered = ecg_low_pass(data, 8, sampling_freq)
	filtered = ecg_high_pass(filtered, 16, sampling_freq)
	# filtered = ecg_bandpass(data, sampling_freq)
	# Derivative
	derivative = scipy.signal.savgol_filter(filtered, 5, 1)
	# Absolute
	filtered = np.abs(derivative)
	# Moving Average (80ms)
	n_signals = data.shape[1]
	mean_size = int(0.08/(1.0/sampling_freq))
	window = np.lib.stride_tricks.sliding_window_view(filtered, mean_size, axis=0)
	filtered = np.mean(window, axis=2)

	# Peak Detection
	peak_window_width = int(0.05 * sampling_freq)
	min_peak_dist = int(0.20 * sampling_freq)
	#signal_peaks = detect_peaks(window_means)
	std = np.std(filtered, axis=0)
	mean = np.mean(filtered, axis=0)
	threshold = mean + 1*std

	# Peak Detection
	signal_peaks = list()
	for i in range(n_signals):
		# Detect the peaks
		fsig = filtered[:,i]
		deriv_sig = derivative[:,i]
		data_sig = data[:,i]
		potential_peaks = scipy.signal.find_peaks(
			fsig,
			#width=peak_window_width, 
			#plateau_size=np.floor_divide(mean_window_size, 2),
			#height=threshold[i]
		)[0]
		# Shift to align with original signal
		#potential_peaks += + (mean_size // 2)
		
		# Refine
		intervals = list(map(lambda x: [], range(data.shape[0] // sampling_freq)))
		for pp in potential_peaks:
			intervals[pp // sampling_freq].append(fsig[pp])
		qrs_peaks = np.zeros(len(intervals))
		noise = []
		for i in range(len(intervals)):
			qrs_peak = max(intervals[i])
			for val in intervals[i]:
				if val != qrs_peak:
					noise.append(val)

		noise_mean = np.mean(noise)
		qrs_peak_mean = np.mean(qrs_peaks)
		threshold = noise_mean+0.3125*(qrs_peak_mean-noise_mean)

		new_peaks = list()
		#shift = mean_size // 2
		lookaround = int(0.2/(1.0/sampling_freq)) # 200ms
		twave_lookaround = int(0.36/(1.0/sampling_freq)) # 360ms
		low_index = 0
		while True:
			high_index = np.searchsorted(potential_peaks, potential_peaks[low_index] + lookaround)
			peak_idx = None
			peak_val = None
			for i in range(low_index, high_index):
				idx = potential_peaks[i]
				val = fsig[idx]
				if val > threshold and (peak_val is None or val > peak_val):
					peak_idx = idx
					peak_val = val

			low_index += 1
			if peak_idx is not None:
				shifted = peak_idx #+ shift
				
				prev = None if len(new_peaks) == 0 else new_peaks[-1]
				if prev != shifted:
					sample_diff = 0 if prev is None else shifted - prev
					can_add = (
						(prev is None) or
						(sample_diff > twave_lookaround) or
						(sample_diff <= twave_lookaround and deriv_sig[shifted] > deriv_sig[prev]/2)
					)
					can_add = True
					if can_add:
						new_peaks.append(shifted)
						
			if high_index >= len(potential_peaks) - 1:
				break
			
		signal_peaks.append(new_peaks)

	fig = go.Figure()
	for signal in range(n_signals):
		fig.add_trace(go.Scatter(x=np.arange(data.shape[0]), y=data[:,signal], mode="lines", name=f"ECG Signal #{signal+1}"))
		fpad = np.pad(filtered, ((mean_size//2,),(0,)), mode="edge")
		fig.add_trace(go.Scatter(x=np.arange(data.shape[0]), y=derivative[:,signal], mode="lines", name=f"Derivated ECG Signal #{signal+1}"))
		fig.add_trace(go.Scatter(x=np.arange(fpad.shape[0]), y=fpad[:,signal], mode="lines", name=f"Filtered ECG Signal #{signal+1}"))
		fig.add_trace(go.Scatter(x=signal_peaks[signal], y=data[signal_peaks[signal],signal], mode="markers", name=f"ECG Peaks #{signal+1}"))

	#fig.add_trace(go.Scatter(x=filtered, mode="lines", name="Filtered ECG"))
	#fig = px.line(data, log_y=True)
	#fig.add_scatter()
	#px.line(filtered, log_y=True).show()
	fig.update_yaxes(type="log")
	fig.show()

	#for i in range(n_signals):
	#	t.add_scatter(x=signal_peaks[i], y=data[signal_peaks[i], i], mode="markers")
	#t.add_scatter(x=signal_peaks, y=data[signal_peaks], mode="markers")
	#t.show()
		
	#print(window)
	#np.unique(np.argmax(window, axis=3), axis=1)
	code.interact(local=locals())
	

def ecg_pan_tompkins(data, sampling_freq):

	pass

ecg_hamilton(X_train[4], 100)

#px.line(ecg_bandpass(X_train[0], 100), title="TETSSET").show()
#px.line(X_train[0], title="TETSSET").show()
