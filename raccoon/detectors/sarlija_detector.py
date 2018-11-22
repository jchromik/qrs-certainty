import sys
sys.path.append("..")

from detectors.nn_detector import NNDetector
from detectors.window_generator import WindowGenerator

from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.utils import multi_gpu_model

import numpy as np

class SarlijaDetector(NNDetector):

    # Initialization

    def __init__(
        self, batch_size, window_size, detection_size,
        epochs=1, gpus=1
    ):
        self.batch_size = batch_size
        self.window_size = window_size
        self.detection_size = detection_size
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __repr__(self):
        return "Sarlija Detector"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tWindow Size: {}".format(self.window_size),
            "\tDetection Size: {}".format(self.detection_size)
        ])

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

    def train(self, ecg_signals, trigger_points):
        self.history = self.model.fit_generator(
            generator = WindowGenerator(
                ecg_signals, self.batch_size, self.window_size, trigger_points,
                self.detection_size, True),
            shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    # NNDetector interface

    def _trigger_signal(self, ecg_signal):
        predictions = self.model.predict_generator(
            generator = WindowGenerator(
                [ecg_signal], self.batch_size, self.window_size,
                wrap_samples=True),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())