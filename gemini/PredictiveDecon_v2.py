# -*- coding: utf-8 -*-
"""

@author: Holger Quintero
"""

import numpy as np
import matplotlib.pyplot as plt

def MyPredictiveDecon(data, n, gap, mu=0.1):
    scal = np.max(data)
    data = data / scal
    
    ndata = np.size(data)
    
    hhorig = np.correlate(data, data, mode='full') 
    nhh = np.size(hhorig)
    if nhh % 2 == 1: 
        nhhmid = int(nhh / 2) + 1
    else:
        nhhmid = int(nhh / 2)
        
    n_minus = int(n / 2) + 1
    n_plus = int(n / 2)

    hh = hhorig[nhhmid-n_minus:nhhmid+n_plus]
    nhh = np.size(hh)
    
    M = np.zeros([n, n])
    if nhh % 2 == 0:
        nhh2 = int(nhh / 2) - 1
    else:
        nhh2 = int(nhh / 2)
        
    for ii in range(n):
        if nhh2 - ii < 0:
            hhadd = np.concatenate([np.zeros(ii - nhh2), hh[max(nhh2 - ii, 0):nhh2 - ii + n]])
        else:
            hhadd = hh[max(nhh2 - ii, 0):nhh2 - ii + n]

        nhhadd = np.size(hhadd)
        M[ii, :nhhadd] = hhadd
        
    M = M * (1 + mu * np.eye(n))
    y = np.zeros(n)
    y[0:n] = hhorig[nhhmid + gap - 1:nhhmid + gap + n - 1]
    f = np.matmul(np.linalg.inv(M), y)
    f2 = np.zeros(n + gap)
    f2[0] = 1.
    f2[gap:] = -f
    hfilt = np.convolve(data, f2)
    hfilt = hfilt[0:ndata] / np.max(hfilt) * scal
        
    return f2, hfilt

# --- 1. Generate Synthetic Seismic Trace ---
np.random.seed(42)
dt = 0.002  # 2 ms sample rate
t = np.arange(0, 1.0, dt)

# Define a Ricker wavelet (standard seismic source wavelet)
def ricker(t, f_peak):
    pi2 = (np.pi * f_peak * t) ** 2
    return (1 - 2 * pi2) * np.exp(-pi2)

wavelet = ricker(np.arange(-0.05, 0.05, dt), f_peak=30)

# Create a sparse reflectivity spike series
reflectivity = np.zeros_like(t)
reflectivity[100] = 1.0   # Primary reflection at t = 0.2s
reflectivity[250] = -0.6  # Second primary reflection at t = 0.5s

# Convolve reflectivity with wavelet to get primary trace
primary_trace = np.convolve(reflectivity, wavelet, mode='same')

# Add a periodic multiple (simulating a seabed bounce delayed by 50 samples / 100ms)
gap_samples = 50
multiple_trace = np.zeros_like(primary_trace)
multiple_trace[gap_samples:] = primary_trace[:-gap_samples] * 0.5  # 50% amplitude echo

# Total synthetic trace with slight random background noise
synthetic_trace = primary_trace + multiple_trace + np.random.normal(0, 0.01, size=len(t))

# --- 2. Apply Predictive Deconvolution ---
n_filter = 20
f2, filtered_trace = MyPredictiveDecon(synthetic_trace, n=n_filter, gap=gap_samples, mu=0.01)

# --- 3. Plotting Results ---
plt.figure(figsize=(12, 6))
plt.plot(t, synthetic_trace, label='Synthetic Trace (with Multiples & Noise)', color='black', alpha=0.7)
plt.plot(t, filtered_trace, label='Filtered Trace (Predictive Decon)', color='crimson', linewidth=1.5)
plt.title('Predictive Deconvolution Test on Synthetic Seismic Trace')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
