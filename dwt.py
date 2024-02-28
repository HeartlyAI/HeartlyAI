from ptb_loader import load_ptb_xl
import pywt
import numpy as np
import plotly.graph_objects as go
 

#print(pywt.families())

# Load ECG data
dataset = load_ptb_xl("./ptb-xl/", "lr")
X_train = dataset["train"][0]
y_train = dataset["train"][1]


# Visualize the ECG data
def visualize_ecg(data, n_signals=1):
    fig = go.Figure()
    for signal in range(n_signals):
        fig.add_trace(go.Scatter(x=np.arange(data.shape[0]), y=data[:,signal], mode="lines", name=f"ECG Signal #{signal+1}"))
    fig.show()



def visualize_signals(data, wavelet='dmey', n_signals=1):
    fig = go.Figure()
    for signal in range(n_signals):
        # Original ECG signal
        fig.add_trace(go.Scatter(x=np.arange(data.shape[0]), y=data[:, signal], mode='lines', name=f'Original ECG Signal #{signal+1}'))
        
        # Perform DWT
        cA, cD = dwt(data[:, signal], wavelet)
        
        # Approximation coefficients
        fig.add_trace(go.Scatter(x=np.arange(len(cA)), y=cA, mode='lines', name=f'Approximation Coefficients #{signal+1}'))
        
        # Detail coefficients
        fig.add_trace(go.Scatter(x=np.arange(len(cD)), y=cD, mode='lines', name=f'Detail Coefficients #{signal+1}'))
    
    fig.update_layout(title='ECG Signal and DWT Results', xaxis_title='Time', yaxis_title='Amplitude')
    fig.show()


# dwt returns coefficients
def dwt(data, wavelet="dmey"):
    cA, cD = pywt.dwt(data, wavelet)
    return cA, cD






visualize_ecg(X_train[4], 12)


