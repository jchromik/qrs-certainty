from keras.utils import Sequence

from ..utils.triggerutils import points_to_signal


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
        if window_index < 0: raise IndexError("Window index negative.")
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
