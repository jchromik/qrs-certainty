from keras.utils import Sequence
import numpy as np

from ..utils.indexutils import index_pairs_for_batch


class WindowGenerator(Sequence):

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

    def __adjust_chunk_index(self, chunk_index):
        if chunk_index in range(len(self.signal_chunks)):
            return chunk_index

        raise IndexError("Chunk index {} out of bounds [{},{}].".format(
            chunk_index, 0, len(self.signal_chunks)))

    def __adjust_window_index(self, chunk_index, window_index):
        chunk = self.signal_chunks[chunk_index]
        usable_chunk_length = len(chunk) - self.window_size + 1
        
        if window_index in range(usable_chunk_length):
            return window_index
        
        if window_index == usable_chunk_length:
            return window_index - 1

        raise IndexError(
            "Window index {} out of bounds [{},{}] in chunk {}.".format(
                window_index, 0, usable_chunk_length, chunk_index))

    def __adjust_indexes(self, chunk_index, window_index):
        """Sometimes we have to deal with off-by-one errors when index pairs are
        rescaled due to e.g. window averaging. This function tries to adjust
        indexes if necessary and possible. Otherwise it throws.
        """
        chunk_index = self.__adjust_chunk_index(chunk_index)
        window_index = self.__adjust_window_index(chunk_index, window_index)
        return chunk_index, window_index

    def index_pairs_for_batch(self, batch_index):
        return index_pairs_for_batch(
            batch_index, self.batch_size, self.window_size,
            chunk_sizes=[len(chunk) for chunk in self.signal_chunks])

    def window(self, chunk_index, window_index):
        chunk_index, window_index = self.__adjust_indexes(chunk_index, window_index)
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
