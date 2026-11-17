# -*- coding: utf-8 -*-
"""

@author: Holger Quintero
"""


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


#%%%

indat=np.load('S30.npz')
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
plt.ylim([9,0])
plt.xlim([30,125])


import numpy as np
import matplotlib.pyplot as plt

# --- 1. Predictive Deconvolution Function ---
def MyPredictiveDecon(data, n, gap, mu=0.1):
    scal = np.max(np.abs(data))  # Using max(abs) prevents sign-flip normalization issues
    data_norm = data / scal
    
    ndata = np.size(data_norm)
    
    hhorig = np.correlate(data_norm, data_norm, mode='full') 
    nhh = np.size(hhorig)
    if nhh % 2 == 1: 
        nhhmid = int(nhh / 2) + 1
    else:
        nhhmid = int(nhh / 2)
        
    n_minus = int(n / 2) + 1
    n_plus = int(n / 2)

    hh = hhorig[nhhmid - n_minus : nhhmid + n_plus]
    nhh = np.size(hh)
    
    M = np.zeros([n, n])
    if nhh % 2 == 0:
        nhh2 = int(nhh / 2) - 1
    else:
        nhh2 = int(nhh / 2)
        
    for ii in range(n):
        if nhh2 - ii < 0:
            hhadd = np.concatenate([np.zeros(ii - nhh2), hh[max(nhh2 - ii, 0) : nhh2 - ii + n]])
        else:
            hhadd = hh[max(nhh2 - ii, 0) : nhh2 - ii + n]

        nhhadd = np.size(hhadd)
        M[ii, :nhhadd] = hhadd
        
    M = M * (1 + mu * np.eye(n))
    y = np.zeros(n)
    y[0:n] = hhorig[nhhmid + gap - 1 : nhhmid + gap + n - 1]
    
    f = np.matmul(np.linalg.inv(M), y)
    f2 = np.zeros(n + gap)
    f2[0] = 1.0
    f2[gap:] = -f
    
    hfilt = np.convolve(data_norm, f2)
    hfilt = hfilt[0:ndata] / np.max(np.abs(hfilt)) * scal
        
    return f2, hfilt

# --- 2. Load Dataset & Define Parameters ---
indat = np.load('S30.npz')
dat = indat['dat']
offset = indat['offset']
ntr, nt = np.shape(dat)

dt = 0.004
time = np.arange(0, nt) * dt

# Filter parameters (adjust gap=300 to test Alison's parameter)
trace_idx = 10
n_len = 1500
gap_samples = 350
mu_val = 0.1

# --- 3. Extract Trace & Apply Filter ---
raw_trace = dat[trace_idx]
filt_op, filt_trace = MyPredictiveDecon(raw_trace, n=n_len, gap=gap_samples, mu=mu_val)

# Normalize both traces by peak absolute amplitude for direct visual comparison
raw_trace_norm = raw_trace / np.max(np.abs(raw_trace))
filt_trace_norm = filt_trace / np.max(np.abs(filt_trace))

# --- 4. Compute Normalized Amplitude Spectra ---
freqs = np.fft.rfftfreq(nt, d=dt)
raw_amp = np.abs(np.fft.rfft(raw_trace))
filt_amp = np.abs(np.fft.rfft(filt_trace))

raw_amp_norm = raw_amp / np.max(raw_amp)
filt_amp_norm = filt_amp / np.max(filt_amp)

# --- 5. Plot Normalized Trace & Amplitude Spectrum ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Normalized Time-Domain Trace
axes[0].plot(time, raw_trace_norm, label='Initial Trace (Normalized)', color='black', alpha=0.65, linewidth=1.2)
axes[0].plot(time, filt_trace_norm, label=f'Filtered Trace (n={n_len}, gap={gap_samples}, mu={mu_val})', color='crimson', linewidth=1.2)
axes[0].set_title(f'Time-Domain Comparison — Trace #{trace_idx} (Offset: {offset[trace_idx]:.2f} km)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Normalized Amplitude')
axes[0].set_xlim([0, 9])
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Panel 2: Normalized Amplitude Spectrum
axes[1].plot(freqs, raw_amp_norm, label='Initial Trace Spectrum', color='black', alpha=0.65, linewidth=1.5)
axes[1].plot(freqs, filt_amp_norm, label='Filtered Trace Spectrum', color='crimson', linewidth=1.5)
axes[1].set_title(f'Normalized Amplitude Spectrum — Trace #{trace_idx}')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Normalized Spectral Amplitude')
axes[1].set_xlim([0, 60])  # Focused on seismic bandwidth (0–60 Hz)
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
