from ..utils.triggerutils import points_to_signal
from keras.utils import Sequence

import numpy as np

class SingleSignalWindowGenerator(Sequence):

    def __init__(
        self, signal_chunks, batch_size, window_size,
        trigger_chunks=[], detection_size=None, wrap_samples=False
    ):
        detection_size = detection_size if detection_size else window_size
        
        self.swg = SignalWindowGenerator(
            signal_chunks, batch_size, window_size, wrap_samples)
        
        self.window_size = window_size
        self.trigger_signals = [
            points_to_signal(trigger, len(chunk), detection_size)
            for trigger, chunk in zip(trigger_chunks, signal_chunks)]

    def _label(self, chunk_index, window_index):
        if not self.swg.valid_chunk(chunk_index):
            raise IndexError("Chunk index out of bounds.")
        
        if not self.swg.valid_window(chunk_index, window_index):
            raise IndexError("Window index out of bounds.")

        if not self.trigger_signals:
            raise RuntimeError("Generator has no labels.")

        trigger_signal = self.trigger_signals[chunk_index]
        return trigger_signal[window_index + self.window_size // 2]

    def _labels(self, index_pairs):
        return [
            self._label(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def window_batch(self, batch_index):
        return self.swg.batch(batch_index)

    def label_batch(self, batch_index):
        return self._labels(self.swg.index_pairs_for_batch(batch_index))

    def train_batch(self, batch_index):
        index_pairs = self.swg.index_pairs_for_batch(batch_index)
        return self.swg.windows(index_pairs), self._labels(index_pairs)

    def __getitem__(self, index):
        windows = self.swg[index]
        if not self.trigger_signals:
            return windows
        
        labels = self.label_batch(index)
        return windows, labels

    def __len__(self):
        return len(self.swg)


class SignalWindowGenerator(Sequence):

    def __init__(
        self, signal_chunks, batch_size, window_size, wrap_samples=False
    ):
        self.signal_chunks = signal_chunks
        self.batch_size = batch_size
        self.window_size = window_size
        self.wrap_samples = wrap_samples

    def valid_chunk(self, chunk_index):
        return chunk_index in range(0, len(self.signal_chunks))

    def valid_window(self, chunk_index, window_index):
        chunk = self.signal_chunks[chunk_index]
        return window_index in range(0, len(chunk) - self.window_size + 1)

    def index_pair(self, window_index):
        for chunk_index, chunk in enumerate(self.signal_chunks):
            usable_chunk_length = len(chunk) - self.window_size
            if window_index <= usable_chunk_length:
                return chunk_index, window_index
            else:
                window_index -= (usable_chunk_length + 1)
        raise IndexError("Window index out of bounds.")

    def index_pairs_for_batch(self, batch_index):
        start = batch_index*self.batch_size
        end = (batch_index+1)*self.batch_size
        return [
            self.index_pair(window_index)
            for window_index in range(start, end)]

    def window(self, chunk_index, window_index):
        if not self.valid_chunk(chunk_index):
            raise IndexError("Chunk index out of bounds.")

        chunk = self.signal_chunks[chunk_index]

        if not self.valid_window(chunk_index, window_index):
            raise IndexError("Window index out of bounds.")

        start = window_index
        end = window_index + self.window_size
        
        return chunk[start:end]

    def windows(self, index_pairs):
        return [
            self.window(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def batch(self, batch_index):
        return self.windows(self.index_pairs_for_batch(batch_index))

    def __getitem__(self, index):
        batch = np.array(self.batch(index))
        if self.wrap_samples:
            batch.shape = (self.batch_size, self.window_size, 1)
        return batch

    def __len__(self):
        num_windows = sum([
            len(chunk) - self.window_size + 1
            for chunk in self.signal_chunks])
        return num_windows // self.batch_size