import numpy as np
import matplotlib.pyplot as plt
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

# --- 1. Predictive Deconvolution Function ---
def MyPredictiveDecon(data, n, gap, mu=0.1):
    scal = np.max(np.abs(data))
    if scal == 0:
        return np.zeros(n + gap), data
    data_norm = data / scal
    
    ndata = np.size(data_norm)
    hhorig = np.correlate(data_norm, data_norm, mode='full') 
    nhh = np.size(hhorig)
    
    nhhmid = (nhh // 2) + 1 if nhh % 2 == 1 else nhh // 2
        
    n_minus = (n // 2) + 1
    n_plus = n // 2

    hh = hhorig[nhhmid - n_minus : nhhmid + n_plus]
    nhh = np.size(hh)
    
    M = np.zeros([n, n])
    nhh2 = (nhh // 2) - 1 if nhh % 2 == 0 else nhh // 2
        
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
    
    try:
        f = np.matmul(np.linalg.inv(M), y)
    except np.linalg.LinAlgError:
        f = np.zeros(n) # Fallback if matrix is singular
        
    f2 = np.zeros(n + gap)
    f2[0] = 1.0
    f2[gap:] = -f
    
    hfilt = np.convolve(data_norm, f2)
    max_hfilt = np.max(np.abs(hfilt))
    if max_hfilt > 0:
        hfilt = hfilt[0:ndata] / max_hfilt * scal
    else:
        hfilt = hfilt[0:ndata] * scal
        
    return f2, hfilt

# --- 2. Load Dataset & Setup Processing ---
indat = np.load('S30.npz')
dat = indat['dat']
offset = indat['offset']
ntr, nt = np.shape(dat)

dt = 0.004
time = np.arange(0, nt) * dt
gval = 1

# Apply gain if gaindat function is available in your environment
# datgain = gaindat(time, dat, gval)

# Choose processing parameters
n_len = 1500
gap_samples = 350
mu_val = 0.1

filtdat = np.zeros_like(dat)

print(f"Processing {ntr} traces with n={n_len}, gap={gap_samples}...")
for ii in range(ntr):
    _, filtdat[ii] = MyPredictiveDecon(dat[ii], n=n_len, gap=gap_samples, mu=mu_val)

# --- 3. Extract Random Trace ---
np.random.seed(42)  # For reproducibility
rand_trace_idx = np.random.randint(0, ntr)
raw_rand = dat[rand_trace_idx]
filt_rand = filtdat[rand_trace_idx]

# Normalize for direct comparison
raw_rand_norm = raw_rand / np.max(np.abs(raw_rand))
filt_rand_norm = filt_rand / np.max(np.abs(filt_rand))

# --- 4. Compute Ensemble-Averaged Amplitude Spectra ---
freqs = np.fft.rfftfreq(nt, d=dt)

# Compute spectra across all traces (axis=1)
raw_spectra = np.abs(np.fft.rfft(dat, axis=1))
filt_spectra = np.abs(np.fft.rfft(filtdat, axis=1))

# Average across all traces (axis=0)
avg_raw_spectrum = np.mean(raw_spectra, axis=0)
avg_filt_spectrum = np.mean(filt_spectra, axis=0)

# Normalize average spectra
avg_raw_norm = avg_raw_spectrum / np.max(avg_raw_spectrum)
avg_filt_spectrum_norm = avg_filt_spectrum / np.max(avg_filt_spectrum)

# --- 5. Comprehensive Multi-Panel Visualization ---
fig = plt.figure(figsize=(16, 12))

# Panel A & B: Complete Dataset (Wiggle Sections)
# Note: Ensure `wiggle` function is imported in your working notebook/script
plt.subplot(2, 2, 1)
wiggle(dat, xvals=offset, scale=1.0, clip=80, sampr=dt)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Offset (km)')
plt.ylabel('Time (s)')
plt.title('Original Dataset (S30)')
plt.ylim([9, 0])
plt.xlim([30, 125])

plt.subplot(2, 2, 2)
wiggle(filtdat, xvals=offset, scale=1.0, clip=80, sampr=dt)
plt.gca().invert_yaxis()
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Offset (km)')
plt.ylabel('Time (s)')
plt.title(f'Filtered Dataset (n={n_len}, gap={gap_samples})')
plt.ylim([9, 0])
plt.xlim([30, 125])

# Panel C: Random Trace Comparison (Time Domain)
plt.subplot(2, 2, 3)
plt.plot(time, raw_rand_norm, label=f'Original Trace #{rand_trace_idx}', color='black', alpha=0.6, linewidth=1)
plt.plot(time, filt_rand_norm, label=f'Filtered Trace #{rand_trace_idx}', color='crimson', linewidth=1.2)
plt.title(f'Random Trace Comparison (Offset: {offset[rand_trace_idx]:.2f} km)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Amplitude')
plt.xlim([0, 9])
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# Panel D: Ensemble-Averaged Amplitude Spectrum (All Traces)
plt.subplot(2, 2, 4)
plt.plot(freqs, avg_raw_norm, label='Avg Initial Spectrum (All Traces)', color='black', alpha=0.6, linewidth=1.5)
plt.plot(freqs, avg_filt_spectrum_norm, label='Avg Filtered Spectrum (All Traces)', color='crimson', linewidth=1.5)
plt.title('Ensemble-Averaged Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Amplitude')
plt.xlim([0, 60])
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()