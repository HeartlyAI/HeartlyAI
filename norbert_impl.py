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
qrs_list = preprocessing.r_peaks_hamilton(data)


fig = go.Figure()
for signal in range(data.shape[0]):
	sqrs = np.array(qrs_list[signal])
	fig.add_trace(go.Scatter(x=np.arange(data.shape[1]), y=data[signal], mode="lines", name=f"ECG Signal #{signal+1}"))
	fig.add_trace(go.Scatter(x=sqrs, y=data[signal,sqrs], mode="markers", name=f"ECG Peaks #{signal+1}"))
fig.show()
