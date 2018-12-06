from ..utils.triggerutils import points_to_signal

class SingleSignalWindowGenerator():

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

    def __index_pair(self, window_index):
        for chunk_index, chunk in enumerate(self.signal_chunks):
            usable_chunk_length = len(chunk) - self.window_size
            if window_index < usable_chunk_length:
                return chunk_index, window_index
            else:
                window_index -= usable_chunk_length
        raise IndexError("Window index out of bounds.")

    def __index_pairs_for_batch(self, batch_index):
        start = batch_index*self.batch_size
        end = (batch_index+1)*self.batch_size
        return [
            self.__index_pair(window_index)
            for window_index in range(start, end)]

    def __window(self, chunk_index, window_index):
        start = window_index*self.window_size
        end = (window_index+1)*self.window_size
        return chunk[start:end]

    def __windows(self, index_pairs):
        return [
            self.__window(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def __label(self, chunk_index, window_index):
        trigger_signal = self.trigger_signals[chunk_index]
        return trigger_signal[window_index + self.window_size // 2]

    def __labels(self, index_pairs):
        return [
            self.__label(chunk_index, window_index)
            for chunk_index, window_index in index_pairs]

    def window_batch(self, batch_index):
        return self.__windows(self.__index_pairs_for_batch(batch_index))

    def label_batch(self, batch_index):
        return self.__labels(self.__index_pairs_for_batch(batch_index))

    def batch(self, batch_index):
        index_pairs = self.__index_pairs_for_batch(batch_index)
        return self.__windows(index_pairs), self.__labels(index_pairs)

    def __getitem__(self, index):
        return batch(index)

    def __len__(self):
        pass