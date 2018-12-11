from keras.utils import Sequence
import numpy as np

from ..utils.triggerutils import points_to_signal


def rescale(index_pairs, old_window_size, new_window_size):
    return [
        (chunk_index, (window_index*new_window_size) // old_window_size)
        for chunk_index, window_index in index_pairs]


class MultiSignalWindowGenerator(Sequence):

    def __init__(
            self, signals, batch_size, window_sizes,
            trigger_chunks=None, detection_size=None, wrap_samples=False
    ):
        self.window_generators = [
            SignalWindowGenerator(signal, batch_size,
                                  window_size, wrap_samples)
            for signal, window_size in zip(signals, window_sizes)]

        self.ref_window_generator = self.window_generators[0]
        self.ref_window_size = window_sizes[0]

        self.labels = (
            LabelGenerator(
                trigger_chunks=trigger_chunks,
                chunk_lengths=[len(chunk) for chunk in signals[0]],
                batch_size=batch_size,
                window_size=window_sizes[0],
                detection_size=detection_size if detection_size else window_sizes[0])
            if trigger_chunks is not None else None)

    def __getitem__(self, index):
        index_pairs = self.ref_window_generator.index_pairs_for_batch(index)

        window_batches = [
            gen.windows(
                rescale(index_pairs, self.ref_window_size, gen.window_size),
                as_array=True)
            for gen in self.window_generators]

        if self.labels is None:
            return window_batches

        labels = self.labels[index]
        return window_batches, labels

    def __len__(self):
        return len(self.window_generators[0])


class SingleSignalWindowGenerator(Sequence):

    def __init__(
            self, signal_chunks, batch_size, window_size,
            trigger_chunks=None, detection_size=None, wrap_samples=False
    ):
        self.windows = SignalWindowGenerator(
            signal_chunks, batch_size, window_size, wrap_samples)
        self.labels = (
            LabelGenerator(
                trigger_chunks=trigger_chunks,
                chunk_lengths=[len(chunk) for chunk in signal_chunks],
                batch_size=batch_size,
                window_size=window_size,
                detection_size=detection_size if detection_size else window_size)
            if trigger_chunks is not None else None)

    def __getitem__(self, index):
        windows = self.windows[index]
        if self.labels is None:
            return windows
        labels = self.labels[index]
        return windows, labels

    def __len__(self):
        return len(self.windows)


class LabelGenerator(Sequence):

    def __init__(
            self, trigger_chunks, chunk_lengths,
            batch_size, window_size, detection_size
    ):
        self.trigger_signals = [
            points_to_signal(
                points=trigger,
                signal_length=chunk_length,
                window_size=detection_size)
            for trigger, chunk_length in zip(trigger_chunks, chunk_lengths)]
        self.chunk_lengths = chunk_lengths
        self.batch_size = batch_size
        self.window_size = window_size

    def __check_index(self, chunk_index, window_index):
        if chunk_index not in range(0, len(self.trigger_signals)):
            raise IndexError("Chunk index out of bounds.")

        chunk_length = self.chunk_lengths[chunk_index]

        if window_index not in range(0, chunk_length - self.window_size + 1):
            raise IndexError("Window index out of bounds.")

    def index_pair(self, window_index):
        for chunk_index, chunk_length in enumerate(self.chunk_lengths):
            usable_chunk_length = chunk_length - self.window_size
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

    def label(self, chunk_index, window_index):
        self.__check_index(chunk_index, window_index)
        trigger_signal = self.trigger_signals[chunk_index]
        return trigger_signal[window_index + self.window_size // 2]

    def labels(self, index_pairs):
        return [
            self.label(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def __getitem__(self, index):
        return self.labels(self.index_pairs_for_batch(index))

    def __len__(self):
        num_labels = sum([
            chunk_length - self.window_size + 1
            for chunk_length in self.chunk_lengths])
        return num_labels // self.batch_size


class SignalWindowGenerator(Sequence):

    def __init__(
            self, signal_chunks, batch_size, window_size, wrap_samples=False
    ):
        self.signal_chunks = signal_chunks
        self.batch_size = batch_size
        self.window_size = window_size
        self.wrap_samples = wrap_samples

    def __as_array(self, windows):
        windows = np.array(windows)
        if self.wrap_samples:
            windows.shape = (len(windows), self.window_size, 1)
        return windows

    def __check_index(self, chunk_index, window_index):
        if chunk_index not in range(0, len(self.signal_chunks)):
            raise IndexError("Chunk index out of bounds.")

        chunk = self.signal_chunks[chunk_index]

        if window_index not in range(0, len(chunk) - self.window_size + 1):
            raise IndexError("Window index out of bounds.")

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
        self.__check_index(chunk_index, window_index)
        chunk = self.signal_chunks[chunk_index]

        start = window_index
        end = window_index + self.window_size

        return chunk[start:end]

    def windows(self, index_pairs, as_array=False):
        windows = [
            self.window(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

        if as_array:
            return self.__as_array(windows)

        return windows

    def batch(self, batch_index, as_array=False):
        return self.windows(self.index_pairs_for_batch(batch_index), as_array)

    def __getitem__(self, index):
        return self.batch(index, as_array=True)

    def __len__(self):
        num_windows = sum([
            len(chunk) - self.window_size + 1
            for chunk in self.signal_chunks])
        return num_windows // self.batch_size
