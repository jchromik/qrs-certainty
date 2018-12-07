from . import NNDetector
from . import SingleSignalWindowGenerator

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import multi_gpu_model

import numpy as np

class GarciaBerdonesDetector(NNDetector):

    # Initialization

    def __init__(self, name, batch_size, window_size, epochs=1, gpus=0):
        self.name = name
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.gpus = gpus
        self.model = self._build_model()

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tBatch Size: {}".format(self.batch_size),
            "\tWindow Size: {}".format(self.window_size),
            "\tTraining Epochs: {}".format(self.epochs),
            "\tNumber of GPUs used: {}".format(self.gpus)])

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

    def train(self, records, triggers):
        ecg_signals = [record.p_signal.T[0] for record in records]
        self.history = self.model.fit_generator(
            generator = SingleSignalWindowGenerator(
                ecg_signals, self.batch_size, self.window_size, triggers),
            shuffle=True, epochs=self.epochs,
            use_multiprocessing=True, workers=16, max_queue_size=16)

    def trigger_signal(self, record):
        ecg_signal = record.p_signal.T[0]
        predictions = self.model.predict_generator(
            generator = SingleSignalWindowGenerator(
                [ecg_signal], self.batch_size, self.window_size),
            use_multiprocessing=True, workers=16, max_queue_size=16)
        return np.append(
            # zero-padding with half window size due to offset
            np.zeros(self.window_size // 2),
            predictions.flatten())
