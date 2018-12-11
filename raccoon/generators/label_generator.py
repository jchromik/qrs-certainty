from keras.utils import Sequence

from ..utils.indexutils import index_pairs_for_batch
from ..utils.triggerutils import points_to_signal


class LabelGenerator(Sequence):

    def __init__(
            self, trigger_chunks, chunk_sizes,
            batch_size, window_size, detection_size
    ):
        self.trigger_signals = [
            points_to_signal(
                points=trigger,
                signal_length=chunk_length,
                window_size=detection_size)
            for trigger, chunk_length in zip(trigger_chunks, chunk_sizes)]
        self.chunk_sizes = chunk_sizes
        self.batch_size = batch_size
        self.window_size = window_size

    def __check_index(self, chunk_index, window_index):
        if chunk_index not in range(0, len(self.trigger_signals)):
            raise IndexError("Chunk index out of bounds.")

        chunk_length = self.chunk_sizes[chunk_index]

        if window_index not in range(0, chunk_length - self.window_size + 1):
            raise IndexError("Window index out of bounds.")

    def index_pairs_for_batch(self, batch_index):
        return index_pairs_for_batch(
            batch_index, self.batch_size, self.window_size, self.chunk_sizes)

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
            for chunk_length in self.chunk_sizes])
        return num_labels // self.batch_size
