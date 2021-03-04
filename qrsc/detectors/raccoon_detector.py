from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import multi_gpu_model

import numpy as np

from . import NNDetector
from ..generators import MultiSignalWindowGenerator
from ..utils.signalutils import window_average


class RaccoonDetector(NNDetector):

    # Initialization

    def __init__(
        self, name, batch_size, window_size, detection_size, winavg_sizes,
        threshold=None, tolerance=None, epochs=1, gpus=0
    ):
        super().__init__(threshold=threshold, tolerance=tolerance)
        self.name = name
        self.batch_size = batch_size
        self.window_size = window_size
        self.detection_size = detection_size
        self.winavg_sizes = winavg_sizes
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tBatch Size: {}".format(self.batch_size),
            "\tWindow Size: {}".format(self.window_size),
            "\tDetection Size: {}".format(self.detection_size),
            "\tWindow Average Sizes: {}".format(self.winavg_sizes),
            "\tThreshold: {}".format(self.threshold),
            "\tTolerance: {}".format(self.tolerance),
            "\tTraining Epochs: {}".format(self.epochs),
            "\tNumber of GPUs used: {}".format(self.gpus)])

    def _build_model(self):
        visibles = [
            Input(shape=(self.window_size // winavg_size, 1))
            for winavg_size in self.winavg_sizes]
        convs = [
            Conv1D(32, kernel_size=3, activation='relu')(visible)
            for visible in visibles]
        mps = [
            MaxPooling1D(pool_size=3, strides=None)(conv)
            for conv in convs]
        flattens = [Flatten()(mp) for mp in mps]

        concat = concatenate(flattens)
        dense1 = Dense(32, activation='relu')(concat)
        dense2 = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=visibles, outputs=dense2)
        if self.gpus > 1:
            model = multi_gpu_model(model, gpus=self.gpus)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    # QRSDetector interface

    def train(self, records, triggers):
        ecg_signals = [record.p_signal.T[0] for record in records]
        signals = [[
                np.ediff1d(window_average(ecg_signal, winavg_size))
                for ecg_signal in ecg_signals]
            for winavg_size in self.winavg_sizes]
        gen = MultiSignalWindowGenerator(
            signals=signals,
            batch_size=self.batch_size,
            window_sizes=[
                self.window_size // winavg_size
                for winavg_size in self.winavg_sizes],
            trigger_chunks=triggers,
            detection_size=self.detection_size,
            wrap_samples=True
        )
        self.history = self.model.fit_generator(
            generator=gen, shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    def trigger_signal(self, record):
        ecg_signal = record.p_signal.T[0]
        signals = [
            [np.ediff1d(window_average(ecg_signal, winavg_size))]
            for winavg_size in self.winavg_sizes]
        predictions = self.model.predict_generator(
            generator = MultiSignalWindowGenerator(
                signals=signals,
                batch_size=self.batch_size,
                window_sizes=[
                    self.window_size // winavg_size
                    for winavg_size in self.winavg_sizes],
                wrap_samples = True),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())
