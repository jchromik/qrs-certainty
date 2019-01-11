from copy import copy

import numpy as np

from .nn_detector import NNDetector
from .xiang_detector import XiangDetector


class MultiDetectorModel:

    def __init__(self, prototype):
        self.prototype = prototype
        self.detectors = []

    def num_detectors(self):
        return len(self.detectors)

    def save(self, path):
        for detector in self.detectors:
            detector.save_model(path)

    def train(self, records, triggers):
        for record, trigger in zip(records, triggers):
            detector = copy(self.prototype)
            detector.name = record.record_name
            detector.train([record], [trigger])
            self.detectors.append(detector)

    def trigger_signals(self, record):
        return [
            detector.trigger_signal(record)
            for detector in self.detectors]


class XiangEnsemble(NNDetector):

    def __init__(
            self, name, batch_size, window_size, detection_size, aux_ratio,
            threshold=None, tolerance=None,
            depth=1, width=32,
            epochs=1, gpus=0
    ):
        self.name = name
        self.batch_size = batch_size
        self.window_size = window_size
        self.detection_size = detection_size
        self.aux_ratio = aux_ratio
        self.threshold = threshold
        self.tolerance = tolerance
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
        model = MultiDetectorModel(
            XiangDetector(
                name="",  # name set later on
                batch_size=self.batch_size,
                window_size=self.window_size,
                detection_size=self.detection_size,
                aux_ratio=self.aux_ratio,
                depth=self.depth,
                width=self.width,
                epochs=self.epochs,
                gpus=self.gpus))
        return model

    def train(self, records, triggers):
        self.model.train(records, triggers)

    def trigger_signal(self, record):
        trigger_signals = np.array(self.model.trigger_signals(record))
        return trigger_signals.sum(axis=0) / self.model.num_detectors()
