from keras.utils import Sequence

from . import LabelGenerator, WindowGenerator
from ..utils.indexutils import rescale


class MultiSignalWindowGenerator(Sequence):

    def __init__(
            self, signals, batch_size, window_sizes,
            trigger_chunks=None, detection_size=None, wrap_samples=False
    ):
        self.window_generators = [
            WindowGenerator(signal, batch_size,
                            window_size, wrap_samples)
            for signal, window_size in zip(signals, window_sizes)]

        self.ref_window_generator = self.window_generators[0]
        self.ref_window_size = window_sizes[0]

        self.labels = (
            LabelGenerator(
                trigger_chunks=trigger_chunks,
                chunk_sizes=[len(chunk) for chunk in signals[0]],
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
