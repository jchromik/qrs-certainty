from . import NNDetector
from ..generators import MultiSignalWindowGenerator
from ..utils.signalutils import window_average

from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import multi_gpu_model

import numpy as np

class XiangDetector(NNDetector):

    # Initialization

    def __init__(
            self, name, batch_size, window_size, detection_size, aux_ratio,
            threshold=None, tolerance=None,
            depth=1, width=32,
            epochs=1, gpus=0
    ):
        super().__init__(threshold=threshold, tolerance=tolerance)
        self.name = name
        self.batch_size = batch_size
        self.window_size = window_size
        self.detection_size = detection_size
        self.aux_ratio = aux_ratio
        self.depth = depth
        self.width = width
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tBatch Size: {}".format(self.batch_size),
            "\tWindow Size: {}".format(self.window_size),
            "\tDetection Size: {}".format(self.detection_size),
            "\tAux Ratio: {}".format(self.aux_ratio),
            "\tThreshold: {}".format(self.threshold),
            "\tTolerance: {}".format(self.tolerance),
            "\tTraining Epochs: {}".format(self.epochs),
            "\tNumber of GPUs used: {}".format(self.gpus)])

    def _build_model(self):
        visible1 = Input(shape=(self.window_size, 1))
        visible2 = Input(shape=(self.window_size // self.aux_ratio, 1))
        prev1 = visible1
        prev2 = visible2
        for _ in range(self.depth):
            conv1 = Conv1D(self.width, kernel_size=3, activation='relu')(prev1)
            conv2 = Conv1D(self.width, kernel_size=3, activation='relu')(prev2)
            mp1 = MaxPooling1D(pool_size=3, strides=None)(conv1)
            mp2 = MaxPooling1D(pool_size=3, strides=None)(conv2)
            prev1 = mp1
            prev2 = mp1
        fl1 = Flatten()(prev1)
        fl2 = Flatten()(prev2)
        concatenate1 = concatenate([fl1, fl2])
        dense1 = Dense(32, activation='relu')(concatenate1)
        dense2 = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[visible1, visible2], outputs=dense2)
        if self.gpus > 1:
            model = multi_gpu_model(model, gpus=self.gpus)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    # QRSDetector interface

    def train(self, records, triggers):
        ecg_signals = [record.p_signal.T[0] for record in records]
        diff_signals = [np.ediff1d(signal) for signal in ecg_signals]
        aux_signals = [
            np.ediff1d(window_average(signal, self.aux_ratio))
            for signal in ecg_signals]
        gen = MultiSignalWindowGenerator(
            signals=[diff_signals, aux_signals],
            batch_size=self.batch_size,
            window_sizes=[
                self.window_size,
                self.window_size // self.aux_ratio],
            trigger_chunks = triggers,
            detection_size = self.detection_size,
            wrap_samples = True
        )
        self.history = self.model.fit_generator(
            generator = gen, shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    def trigger_signal(self, record):
        ecg_signal = record.p_signal.T[0]
        predictions = self.model.predict_generator(
            generator = MultiSignalWindowGenerator(
                signals=[
                    [np.ediff1d(ecg_signal)],
                    [np.ediff1d(window_average(ecg_signal, self.aux_ratio))]],
                batch_size=self.batch_size,
                window_sizes=[
                    self.window_size,
                    self.window_size // self.aux_ratio],
                wrap_samples = True),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())
