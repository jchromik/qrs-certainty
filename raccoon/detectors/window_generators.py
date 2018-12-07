from ..utils.triggerutils import points_to_signal
from keras.utils import Sequence

import numpy as np

class SingleSignalWindowGenerator(Sequence):

    def __init__(
        self, signal_chunks, batch_size, window_size,
        trigger_chunks=[], detection_size=None
    ):
        detection_size = detection_size if detection_size else window_size

        self.signal_chunks = signal_chunks
        self.batch_size = batch_size
        self.window_size = window_size
        self.trigger_signals = [
            points_to_signal(trigger, len(chunk), detection_size)
            for trigger, chunk in zip(trigger_chunks, signal_chunks)]

    def __check_chunk_index(self, chunk_index):
        if chunk_index not in range(0, len(self.signal_chunks)):
            raise IndexError("Chunk index out of bounds.")

    def __check_window_index(self, chunk, window_index):
        if window_index not in range(0, len(chunk) - self.window_size + 1):
            raise IndexError("Window index out of bounds.")

    def _index_pair(self, window_index):
        for chunk_index, chunk in enumerate(self.signal_chunks):
            usable_chunk_length = len(chunk) - self.window_size
            if window_index <= usable_chunk_length:
                return chunk_index, window_index
            else:
                window_index -= (usable_chunk_length + 1)
        raise IndexError("Window index out of bounds.")

    def _index_pairs_for_batch(self, batch_index):
        start = batch_index*self.batch_size
        end = (batch_index+1)*self.batch_size
        return [
            self._index_pair(window_index)
            for window_index in range(start, end)]

    def _window(self, chunk_index, window_index):
        self.__check_chunk_index(chunk_index)
        chunk = self.signal_chunks[chunk_index]
        self.__check_window_index(chunk, window_index)

        start = window_index
        end = window_index + self.window_size
        
        return chunk[start:end]

    def _windows(self, index_pairs):
        return [
            self._window(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def _label(self, chunk_index, window_index):
        self.__check_chunk_index(chunk_index)
        chunk = self.signal_chunks[chunk_index]
        self.__check_window_index(chunk, window_index)
        if not self.trigger_signals:
            raise RuntimeError("Generator has no labels.")

        trigger_signal = self.trigger_signals[chunk_index]
        return trigger_signal[window_index + self.window_size // 2]

    def _labels(self, index_pairs):
        return [
            self._label(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def window_batch(self, batch_index):
        return self._windows(self._index_pairs_for_batch(batch_index))

    def label_batch(self, batch_index):
        return self._labels(self._index_pairs_for_batch(batch_index))

    def train_batch(self, batch_index):
        index_pairs = self._index_pairs_for_batch(batch_index)
        return self._windows(index_pairs), self._labels(index_pairs)

    def __getitem__(self, index):
        if not self.trigger_signals:
            return np.array(self.window_batch(index))

        windows, labels = self.train_batch(index)
        return np.array(windows), np.array(labels)

    def __len__(self):
        num_windows = sum([
            len(chunk) - self.window_size + 1
            for chunk in self.signal_chunks])
        return num_windows // self.batch_size