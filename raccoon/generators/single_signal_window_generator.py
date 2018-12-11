from keras.utils import Sequence

from . import LabelGenerator, WindowGenerator


class SingleSignalWindowGenerator(Sequence):

    def __init__(
            self, signal_chunks, batch_size, window_size,
            trigger_chunks=None, detection_size=None, wrap_samples=False
    ):
        self.windows = WindowGenerator(
            signal_chunks, batch_size, window_size, wrap_samples)
        self.labels = (
            LabelGenerator(
                trigger_chunks=trigger_chunks,
                chunk_sizes=[len(chunk) for chunk in signal_chunks],
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
