from triggerutils import points_to_signal

from itertools import chain
from keras.utils import Sequence

import numpy as np

# tpoint  ... trigger point
# tstream ... trigger stream

class WindowGenerator(Sequence):

    """Generates windows for Detector ANNs.
    
    Args:
        signal_list (list): A list of ECG signals the windows are generated from.
        batch_size (int): Specifies how many windows (and labels) are contained in one batch.
        window_size (int): Specifies how many sample points are contained in one window.
            This has to be the same size as the NN's input layer.
        tpoints_list (list): A list of trigger points the labels are generated from.
        detection_size (list): Specifies an area in the window.
            The label is 1 if a trigger point is contained in this area.
    """

    # instance creation

    def __init__(
        self, signal_list, batch_size, window_size,
        tpoints_list=None, detection_size=None, wrap_samples=False,
        aux_signal_list=None, aux_ratio=1
    ):
        self.signal_list = signal_list
        self.batch_size = batch_size
        self.window_size = window_size
        self.window_shape = (
             (batch_size, window_size) if not wrap_samples
             else (batch_size, window_size, 1))
        self.detection_size = (
            window_size if detection_size is None
            else detection_size)
        self.tstream_list = self.__tstream_list(signal_list, tpoints_list, self.detection_size)
        self.aux_signal_list = aux_signal_list
        self.aux_ratio = aux_ratio
        self.aux_window_shape = (
             (batch_size, window_size // aux_ratio) if not wrap_samples
             else (batch_size, window_size // aux_ratio, 1))

    def __tstream_list(self, signal_list, tpoints_list, detection_size):
        if tpoints_list is None: return None
        return [
            points_to_signal(tpoints, len(signal), detection_size)
            for signal, tpoints in zip(signal_list, tpoints_list)]

    # Sequence interface

    def __len__(self):
        return sum([len(signal) - self.window_size for signal in self.signal_list]) // self.batch_size

    def __getitem__(self, idx):
        idx_pairs = [
            self.__idx_pair(idx)
            for idx in range(idx*self.batch_size, (idx+1)*self.batch_size)]

        windows = self.__windows(idx_pairs)
        if self.tstream_list is None:
            return windows

        labels = self.__labels(idx_pairs)
        return windows, labels

    # data generation helpers

    def __windows(self, idx_pairs):
        signal_windows = self.__signal_windows(idx_pairs)
        if self.aux_signal_list is None:
            return signal_windows
        aux_windows = self.__aux_windows(idx_pairs)
        return [signal_windows, aux_windows]

    def __signal_windows(self, idx_pairs):
        windows = np.fromiter(
            chain.from_iterable(
                self.__signal_window(idx_pair) for idx_pair in idx_pairs
            ), 'f')
        return windows.reshape(self.window_shape)
    
    def __aux_windows(self, idx_pairs):
        windows = np.fromiter(
            chain.from_iterable(
                self.__aux_window(idx_pair) for idx_pair in idx_pairs
            ), 'f')
        return windows.reshape(self.aux_window_shape)

    def __signal_window(self, idx_pair):
        signal_idx, window_idx = idx_pair
        return self.signal_list[signal_idx][window_idx:window_idx+self.window_size]

    def __aux_window(self, idx_pair):
        signal_idx, window_idx = idx_pair
        aux_signal = self.aux_signal_list[signal_idx]
        window_begin = window_idx // self.aux_ratio
        window_end = (window_idx + self.window_size) // self.aux_ratio
        return aux_signal[window_begin:window_end]

    def __labels(self, idx_pairs):
        return np.fromiter((self.__label(idx_pair) for idx_pair in idx_pairs), 'f')

    def __label(self, idx_pair):
        signal_idx, window_idx = idx_pair
        return self.tstream_list[signal_idx][window_idx + self.window_size // 2]

    def __idx_pair(self, idx):
        """Converts a single continuous index to a pair of signal index
        (specifying which signal to use) and window index (specifying the
        window in the aforementioned signal).  
        """
        signal_idx = 0
        window_idx = idx
        for signal in self.signal_list:
            usable_signal_length = len(signal) - self.window_size
            if window_idx >= usable_signal_length:
                signal_idx += 1
                window_idx -= usable_signal_length
            else: break
        return signal_idx, window_idx