import numpy as np
import time
from utils.onnx import onnx_predict

shape = (1, 1000, 12)

# random data
X = np.random.rand(*shape)

start_time = time.time()
preds = onnx_predict(X, "../model.onnx")
end_time = time.time()

print(f"Prediction took {end_time - start_time} seconds")