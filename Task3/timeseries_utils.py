import numpy as np
import pandas as pd

def rolling_mean(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def rolling_var(arr, window):
    mean = rolling_mean(arr, window)
    return np.array([np.var(arr[i:i+window]) for i in range(len(arr)-window+1)])

def ewma(arr, alpha=0.3):
    s = [arr[0]]
    for x in arr[1:]:
        s.append(alpha*x + (1-alpha)*s[-1])
    return np.array(s)

def fft_bandpass(arr, low, high, fs=1.0):
    f = np.fft.rfftfreq(len(arr), d=1/fs)
    fft_vals = np.fft.rfft(arr)
    fft_vals[(f<low) | (f>high)] = 0
    return np.fft.irfft(fft_vals)
