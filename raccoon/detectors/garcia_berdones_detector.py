import sys
sys.path.append("..")

from detectors.nn_detector import NNDetector
from detectors.window_generator import WindowGenerator

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import multi_gpu_model

import numpy as np

class GarciaBerdonesDetector(NNDetector):

    # Initialization

    def __init__(self, batch_size, window_size, epochs=1, gpus=1):
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __repr__(self):
        return "Garcia-Berdones Detector"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tWindow Size: {}".format(self.window_size)
        ])

    def _build_model(self):
        model = Sequential()
        model.add(Dense(
            self.window_size, activation='relu',
            input_shape=(self.window_size,)))
        model.add(Dense(2 * self.window_size, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if self.gpus > 1:
            model = multi_gpu_model(model, gpus=self.gpus)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model

    # QRSDetector interface

    def train(self, ecg_signals, trigger_points):
        self.history = self.model.fit_generator(
            generator = WindowGenerator(
                ecg_signals, self.batch_size, self.window_size, trigger_points),
            shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    # NNDetector interface

    def _trigger_signal(self, ecg_signal):
        predictions = self.model.predict_generator(
            generator = WindowGenerator(
                [ecg_signal], self.batch_size, self.window_size),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())
