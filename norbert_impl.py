from ptb_loader import load_ptb_xl
import plotly.graph_objects as go
import scipy
import numpy as np
import code
import preprocessing
import time

dataset = load_ptb_xl("./ptb-xl/", "lr")
X_train = dataset.x("train", False)
X_train_digital = dataset.x("train", True)
y_train = dataset.y("train")


print(X_train_digital.shape)
#start = time.monotonic()
#l = preprocessing.r_peaks_hamilton(X_train_digital)

#print(f"Preprocessed in {time.monotonic() - start}s")

data = X_train_digital[4]
signal = 0
qrs_list = preprocessing.qrs_hamilton(data)


fig = go.Figure()
for signal in range(data.shape[0]):
	q = np.array(qrs_list[signal][0])
	r = np.array(qrs_list[signal][1])
	s = np.array(qrs_list[signal][2])

	fig.add_trace(go.Scatter(x=np.arange(data.shape[1]), y=data[signal], mode="lines", name=f"ECG Signal #{signal+1}"))
	fig.add_trace(go.Scatter(x=q, y=data[signal,q], mode="markers", name=f"Q Signal #{signal+1}"))
	fig.add_trace(go.Scatter(x=r, y=data[signal,r], mode="markers", name=f"R Signal #{signal+1}"))
	fig.add_trace(go.Scatter(x=s, y=data[signal,s], mode="markers", name=f"S Signal #{signal+1}"))
fig.show()
