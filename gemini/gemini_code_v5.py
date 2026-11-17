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


#%%

import matplotlib.pyplot as plt
import numpy as np

class fftsig:

    def __init__(self):
        self.f=np.array
        self.FT=np.array
        self.amp=np.array
        self.phase=np.array

def myfft(t,s1):
#mfft(t,s1) computes the Fourier Transform, 
#the phase spectrum and the amplitude spectrum as well as the frequency sampling vector of an input signal
#outputs: 
# f -- the frequency vectory, shifted so that f=0 is in the centre of the array
# s1f -- the FFT of the signal shifted so that f=0 is at the centre of the array
# s1f_amp -- the amplitude spectrum 
# s1f_phase -- the phase spectrum
    nt=np.size(t)
    dt=t[2]-t[1]
    mysig=fftsig()
    s1f=np.fft.fftshift(np.fft.fft(s1))
    mysig.FT=s1f
    s1f_amp=np.abs(s1f)
    mysig.amp=s1f_amp
    s1f_phase=np.arctan2(np.imag(s1f),np.real(s1f))
    mysig.phase=s1f_phase
    f=np.fft.fftshift(np.fft.fftfreq(nt,dt))
    mysig.f=f
    
    
    return mysig

def gaindat(time, data,tpow):
#apply a simple geometrical spreading correction via data*t**tpow
    [ntr,nt]=np.shape(data)
    timemat=np.matmul(np.ones([ntr,1]),time.reshape([1,nt])**tpow)
    datgain=dat*timemat

    return datgain

def clipsign (value, clip):
  clipthese = abs(value) > clip
  return value * ~clipthese + np.sign(value)*clip*clipthese

def wiggle (traces, skipt=1,scale=1.,lwidth=.1,offsets=None,redvel=0., manthifts=None, tshift=0.,sampr=1.,clip=10., dx=1., color='black',fill=True,line=True, ax=None,xvals=None):

  ns = traces.shape[1]
  ntr = traces.shape[0]
  t = np.arange(ns)*sampr
  timereduce = lambda offsets, redvel, shift: [float(offset) / redvel + shift for offset in offsets]

  if (offsets is not None):
    shifts = timereduce(offsets, redvel, tshift)
  elif (manthifts is not None):
    shifts = manthifts
  else:
    shifts = np.zeros((ntr,))

  for i in range(0, ntr, skipt):
    trace = traces[i].copy()
    trace[0] = 0
    trace[-1] = 0

    if xvals is None:
        xval=i*dx
    else:
        xval=xvals[i]
    
    if ax == None:
      if (line):
        plt.plot(xval + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        plt.fill(xval + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)
    else:
      if (line):
        ax.plot(xval + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        ax.fill(xval + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)


#%%
indat=np.load('S8.npz')
dat=indat['dat']
offset=indat['offset']
[ntr,nt]=np.shape(dat)
time=np.arange(0,nt)*0.004
gval=1
datgain=gaindat(time,dat,gval)
plt.figure(figsize=[8,8])
wiggle(datgain,xvals=offset,scale=1.0,clip=80,sampr=0.004)#,offsets=offsettest,redvel=1,shifts=0)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top') 
plt.xlabel('offset (km)')
plt.ylabel('time (s)')
plt.title('Original Data')
plt.ylim([5,0])
plt.xlim([8,105])


#%%

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

# 1. Load the S8.npz dataset using your workflow
indat = np.load("S8.npz")
dat = indat["dat"]
offset = indat["offset"]

[ntr, nt] = np.shape(dat)
time = np.arange(0, nt) * 0.004
gval = 1
datgain = gaindat(time, dat, gval)

# 2. Apply Spiking Deconvolution Trace-by-Trace
# Note: Depending on whether 'dat' shape is [ntr, nt] or [nt, ntr],
# we loop over traces (columns/rows depending on convention).
# Assuming shape is [ntr, nt] where ntr = number of traces, nt = samples per trace:
filter_length = 35
stabilization = 0.01

datgain_decon = np.zeros_like(datgain)

for i in range(ntr):
  trace = datgain[i, :]  # Extract individual trace

  # Compute autocorrelation
  autocorr = np.correlate(trace, trace, mode="full")
  r_pos = autocorr[nt - 1 : nt - 1 + filter_length]

  # Build Toeplitz matrix
  R_mat = toeplitz(r_pos[:filter_length])
  g = np.zeros(filter_length)
  g[0] = 1.0  # Desired spike at lag 0

  # Pre-whitening stabilization
  R_mat[np.diag_indices_from(R_mat)] += stabilization * R_mat[0, 0]

  try:
    filter_weights = np.linalg.solve(R_mat, g)
    datgain_decon[i, :] = np.convolve(trace, filter_weights, mode="same")
  except np.linalg.LinAlgError:
    datgain_decon[i, :] = trace

# 3. Visualize Original Data with Your Wiggle Setup
plt.figure(figsize=[8, 8])
wiggle(datgain, xvals=offset, scale=1.0, clip=80, sampr=0.004)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xlabel("offset (km)")
plt.ylabel("time (s)")
plt.title("Original Data")
plt.ylim([5, 0])
plt.xlim([8, 105])
plt.tight_layout()
plt.show()

# 4. Visualize Deconvoluted Data with Your Wiggle Setup
plt.figure(figsize=[8, 8])
wiggle(datgain_decon, xvals=offset, scale=1.0, clip=80, sampr=0.004)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xlabel("offset (km)")
plt.ylabel("time (s)")
plt.title("Deconvoluted Data (Spiking Decon)")
plt.ylim([5, 0])
plt.xlim([8, 105])
plt.tight_layout()
plt.show()
