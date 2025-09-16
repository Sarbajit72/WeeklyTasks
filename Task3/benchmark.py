import time
import numpy as np
import pandas as pd
from timeseries_utils import rolling_mean, rolling_var, ewma, fft_bandpass

arr = np.random.randn(1000000)

start = time.time()
_ = rolling_mean(arr, 100)
end = time.time()
print("Rolling mean time:", end-start)

start = time.time()
_ = rolling_var(arr, 100)
end = time.time()
print("Rolling var time:", end-start)

start = time.time()
_ = ewma(arr)
end = time.time()
print("EWMA time:", end-start)

start = time.time()
_ = fft_bandpass(arr, 0.01, 0.1)
end = time.time()
print("FFT band-pass time:", end-start)
