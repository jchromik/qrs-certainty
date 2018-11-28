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

    def __init__(self, window_size):
        self.window_size = window_size

    def __repr__(self):
        return "Pan-Tompkins Detector"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tMoving Window Size: {}".format(self.window_size),
            "\tButterworth Bandpass Filter Lowcut: {}".format(LOWCUT),
            "\tButterworth Bandpass Filter Highcut: {}".format(HIGHCUT),
            "\tButterworth Bandpass Filter Order: {}".format(BUTTER_ORDER),
            "\tPeak Detection Threshold: {}".format(IDX_THRESHOLD)])

    # QRSDetector interface

    def trigger_signals(self, records):
        return [self.__pt_signal(record) for record in records]

    def detect(self, records):
        return [
            self.__pt_indexes(self.__pt_signal(record))
            for record in records]

    def triggers_and_signals(self, records):
        trigger_signals = self.trigger_signals(records)
        return (
            trigger_signals,
            [self.__pt_indexes(ts) for ts in trigger_signals])

    # Private

    def __pt_signal(self, record):
        signal = self.__bandpass_filter(record.p_signal.T[0], record.fs)
        signal = np.ediff1d(signal)
        signal = signal ** 2
        return np.convolve(signal, np.ones(self.window_size), CONVOLVE_MODE)

    def __pt_indexes(self, pt_signal):
        return pu.indexes(pt_signal, thres=IDX_THRESHOLD, min_dist=self.window_size)

    def __bandpass_filter(self, signal, fs):
        nyquist_freq = fs / 2
        low = LOWCUT / nyquist_freq
        high = HIGHCUT / nyquist_freq
        b, a = butter(BUTTER_ORDER, [low, high], btype='band')
        return lfilter(b, a, signal)