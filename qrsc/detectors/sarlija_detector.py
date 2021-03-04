from . import NNDetector
from ..generators import SingleSignalWindowGenerator, WindowGenerator

from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.utils import multi_gpu_model

import numpy as np

class SarlijaDetector(NNDetector):

    # Initialization

    def __init__(
            self, name, batch_size, window_size, detection_size,
            threshold=None, tolerance=None, epochs=1, gpus=0
    ):
        super().__init__(threshold=threshold, tolerance=tolerance)
        self.name = name
        self.batch_size = batch_size
        self.window_size = window_size
        self.detection_size = detection_size
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tBatch Size: {}".format(self.batch_size),
            "\tWindow Size: {}".format(self.window_size),
            "\tDetection Size: {}".format(self.detection_size),
            "\tThreshold: {}".format(self.threshold),
            "\tTolerance: {}".format(self.tolerance),
            "\tTraining Epochs: {}".format(self.epochs),
            "\tNumber of GPUs used: {}".format(self.gpus)])

    def _build_model(self):
        model = Sequential()
        model.add(Dropout(0.5, input_shape=(self.window_size, 1)))
        model.add(Conv1D(32, 5, activation='relu'))
        model.add(MaxPooling1D(3, strides=2))
        model.add(Dropout(0.5))
        model.add(Conv1D(32, 5, activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if self.gpus > 1:
            model = multi_gpu_model(model, gpus=self.gpus)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    # QRSDetector interface

    def train(self, records, triggers):
        ecg_signals = [record.p_signal.T[0] for record in records]
        self.history = self.model.fit_generator(
            generator = SingleSignalWindowGenerator(
                ecg_signals, self.batch_size, self.window_size, triggers,
                self.detection_size, True),
            shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    def trigger_signal(self, record):
        ecg_signal = record.p_signal.T[0]
        predictions = self.model.predict_generator(
            generator = WindowGenerator(
                [ecg_signal], self.batch_size, self.window_size,
                wrap_samples=True),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())