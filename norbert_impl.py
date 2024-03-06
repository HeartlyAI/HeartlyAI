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
start = time.monotonic()
l = preprocessing.r_peaks_hamilton(X_train_digital)

print(f"Preprocessed in {time.monotonic() - start}s")
