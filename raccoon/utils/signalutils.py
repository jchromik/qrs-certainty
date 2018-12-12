import numpy as np

def window_average(signal, window_size):
    """Cut signal in windows of size window_size and compute average (mean) for
    each window, thus generating a new signal. The resulting signal is by factor
    window_size smaller than the original signal.
    """
    cutoff = len(signal) % window_size
    size_corrected_signal = signal if cutoff == 0 else signal[:-cutoff]
    return np.mean(size_corrected_signal.reshape(-1, window_size), axis=1)
