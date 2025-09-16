# Task3: High-Performance Time Series Transformation

## Description
This task implements high-performance routines for large time-series data (>1 million rows):
- Rolling window statistics (mean, variance)
- Exponentially weighted moving average (EWMA)
- FFT-based band-pass filtering
- Benchmarking NumPy vs pandas built-ins

## Files
- `timeseries_utils.py` : Functions for rolling stats, EWMA, FFT
- `benchmark.py` : Script to test performance on large arrays

## How to Run
1. Install requirements:
   pip install numpy pandas
2. Run benchmark:
   python benchmark.py
3. Results will show timing for each method.
