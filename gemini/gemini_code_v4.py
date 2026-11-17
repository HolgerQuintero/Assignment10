# -*- coding: utf-8 -*-
"""

@author: Holger Quintero
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

# 1. Generate Synthetic Seismic Data (3 Reflectors)
np.random.seed(42)
n_samples = 600
dt = 0.002  # 2 ms sample rate
time = np.arange(n_samples) * dt

# Create a sparse reflectivity series with exactly 3 reflectors
reflectivity = np.zeros(n_samples)
spike_indices = [150, 300, 450]
spike_amplitudes = [1.0, -0.7, 0.8]
for idx, amp in zip(spike_indices, spike_amplitudes):
  reflectivity[idx] = amp


# Define a causal, minimum-phase wavelet (e.g., damped sine wave starting at t=0)
def minimum_phase_wavelet(f, length, dt):
  t = np.arange(0, length, dt)
  wavelet = t * np.exp(-35 * t) * np.sin(2 * np.pi * f * t)
  wavelet /= np.max(np.abs(wavelet))  # Normalize amplitude
  return wavelet, t


wavelet, t_wavelet = minimum_phase_wavelet(f=35, length=0.1, dt=dt)

# Convolve reflectivity with wavelet to get noise-free seismic trace
seismic_clean = np.convolve(reflectivity, wavelet, mode="same")

# Add random Gaussian noise (1% noise)
noise_level = 0.01
noise = np.random.normal(0, noise_level, size=n_samples)
seismic_noisy = seismic_clean + noise

# 2. Design Spiking Deconvolution for 3 Different Filter Lengths
filter_lengths = [15, 35, 65]
deconvoluted_results = {}

autocorr = np.correlate(seismic_noisy, seismic_noisy, mode="full")

for f_len in filter_lengths:
  r_pos = autocorr[n_samples - 1 : n_samples - 1 + f_len]
  R_mat = toeplitz(r_pos[:f_len])
  g = np.zeros(f_len)
  g[0] = 1.0  # Desired spike at lag 0

  # Pre-whitening stabilization
  stabilization = 0.01 * R_mat[0, 0]
  R_mat[np.diag_indices_from(R_mat)] += stabilization
  filter_weights = np.linalg.solve(R_mat, g)

  # Apply filter
  deconvoluted_results[f_len] = np.convolve(
      seismic_noisy, filter_weights, mode="same"
  )

# 3. Plot Results
plt.figure(figsize=(12, 12))

plt.subplot(5, 1, 1)
plt.stem(time, reflectivity, basefmt="k-", linefmt="k-", markerfmt="ko")
plt.title("1. True Reflectivity Series (3 Reflectors)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)

plt.subplot(5, 1, 2)
plt.plot(t_wavelet, wavelet, "b")
plt.title("2. Minimum-Phase Source Wavelet (35 Hz)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.subplot(5, 1, 3)
plt.plot(
    time,
    deconvoluted_results[15],
    color="darkorange",
    linewidth=1.2,
    label="Filter Length = 15",
)
plt.title("3. Deconvoluted Trace (Filter Length = 15)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.subplot(5, 1, 4)
plt.plot(
    time,
    deconvoluted_results[35],
    color="purple",
    linewidth=1.2,
    label="Filter Length = 35",
)
plt.title("4. Deconvoluted Trace (Filter Length = 35)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.subplot(5, 1, 5)
plt.plot(
    time,
    deconvoluted_results[65],
    color="red",
    linewidth=1.2,
    label="Filter Length = 65",
)
plt.title("5. Deconvoluted Trace (Filter Length = 65)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(time[0], time[-1])
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
