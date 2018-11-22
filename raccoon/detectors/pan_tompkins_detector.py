import sys
sys.path.append("..")

from detectors.non_nn_detector import NonNNDetector

from scipy.signal import butter, lfilter

import numpy as np
import peakutils as pu

LOWCUT = 5
HIGHCUT = 15
BUTTER_ORDER = 1
CONVOLVE_MODE = 'same'
IDX_THRESHOLD = 0.3

class PanTompkinsDetector(NonNNDetector):

    # Initialization

    def __init__(self, signal_freq, moving_window_size):
        self.signal_freq = signal_freq
        self.moving_window_size = moving_window_size

    def __repr__(self):
        return "Pan-Tompkins Detector"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tMoving Window Size: {}".format(self.moving_window_size)
        ])

    # QRSDetector interface

    def trigger_signals(self, ecg_signals):
        return [self.__pt_signal(signal) for signal in ecg_signals]

    def detect(self, ecg_signals):
        return [
            self.__pt_indexes(self.__pt_signal(signal))
            for signal in ecg_signals]

    def triggers_and_signals(self, ecg_signals):
        trigger_signals = self.trigger_signals(ecg_signals)
        return (
            trigger_signals,
            [self.__pt_indexes(ts) for ts in trigger_signals])

    # Private

    def __pt_signal(self, ecg_signal):
        signal = self.__bandpass_filter(ecg_signal)
        signal = np.ediff1d(signal)
        signal = signal ** 2
        return np.convolve(signal, np.ones(self.moving_window_size), CONVOLVE_MODE)

    def __pt_indexes(self, pt_signal):
        return pu.indexes(pt_signal, thres=IDX_THRESHOLD, min_dist=self.moving_window_size)

    def __bandpass_filter(self, signal):
        nyquist_freq = self.signal_freq / 2
        low = LOWCUT / nyquist_freq
        high = HIGHCUT / nyquist_freq
        b, a = butter(BUTTER_ORDER, [low, high], btype='band')
        return lfilter(b, a, signal)