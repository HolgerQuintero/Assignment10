# -*- coding: utf-8 -*-
"""

@author: Holger Quintero
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

# 1. Generate Synthetic Seismic Data
np.random.seed(42)
n_samples = 600
dt = 0.002  # 2 ms sample rate
time = np.arange(n_samples) * dt

# Create a sparse reflectivity series (earth model)
reflectivity = np.zeros(n_samples)
spike_indices = [120, 200, 280, 350, 450, 520]
spike_amplitudes = [1.0, -0.7, 0.85, -0.5, 0.6, -0.8]
for idx, amp in zip(spike_indices, spike_amplitudes):
  reflectivity[idx] = amp


# Define a Ricker wavelet and return its local time axis
def ricker_wavelet(f, length, dt):
  t = np.arange(-length / 2, length / 2, dt)
  pi2_f2_t2 = (np.pi * f * t) ** 2
  wavelet = (1.0 - 2.0 * pi2_f2_t2) * np.exp(-pi2_f2_t2)
  return wavelet, t


wavelet, t_wavelet = ricker_wavelet(f=35, length=0.12, dt=dt)

# Convolve reflectivity with wavelet to get noise-free seismic trace
seismic_clean = np.convolve(reflectivity, wavelet, mode="same")

# Add random Gaussian noise
noise_level = 0.02
noise = np.random.normal(0, noise_level, size=n_samples)
seismic_noisy = seismic_clean + noise

# 2. Design Spiking Deconvolution Filter (Wiener-Levinson / Toeplitz)
filter_length = 35
# Compute autocorrelation of the noisy seismic trace
autocorr = np.correlate(seismic_noisy, seismic_noisy, mode="full")
r_pos = autocorr[n_samples - 1 : n_samples - 1 + filter_length]

# Build autocorrelation Toeplitz matrix and right-hand side vector
R_mat = toeplitz(r_pos[:filter_length])
g = np.zeros(filter_length)
g[0] = 1.0  # Desired output is a spike at t = 0

# Solve R_mat * f = g with pre-whitening stabilization
stabilization = 0.001 * R_mat[0, 0]
R_mat[np.diag_indices_from(R_mat)] += stabilization
filter_weights = np.linalg.solve(R_mat, g)

# 3. Apply Filter
deconvolved = np.convolve(seismic_noisy, filter_weights, mode="same")

# 4. Plot Results
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.stem(time, reflectivity, basefmt="k-", linefmt="k-", markerfmt="ko")
plt.title("1. True Reflectivity Series (Sparse Spikes)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(t_wavelet, wavelet, "b")
plt.title("2. Source Wavelet (Ricker Wavelet, 35 Hz)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(time, seismic_noisy, "k", linewidth=1.2)
plt.title("3. Synthetic Seismic Trace (Convolved + 2% Noise)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(time, deconvolved, "r", linewidth=1.2)
plt.title("4. Deconvoluted Trace (Wavelet Compressed to Spikes)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
