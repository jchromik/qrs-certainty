from abc import abstractmethod
from os.path import dirname
from os import makedirs

from . import QRSDetector
from ..utils.triggerutils import signal_to_points

DEFAULT_THRESHOLD = .8
DEFAULT_TOLERANCE = 10

class NNDetector(QRSDetector):

    def __init__(self, threshold=None, tolerance=None):
        self.threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        self.tolerance = DEFAULT_TOLERANCE if tolerance is None else tolerance

    # Additional abstract method

    @abstractmethod
    def _build_model(self):
        """Build the detector-specific neural network (model)."""
        pass

    # Common implementations

    def reset(self):
        """Rebuild model from scratch throwing away all weights."""
        self.model = self._build_model()

    def save_model(self, path):
        """Save trained model with weights to file."""
        makedirs(dirname(path), exist_ok=True)
        self.model.save(path)

    def trigger(self, record):
        """Find trigger points in single ECG recording."""
        return signal_to_points(
            signal=self.trigger_signal(record),
            tolerance=self.tolerance,
            threshold=self.threshold)

    def trigger_and_signal(self, record):
        """Return trigger signal and trigger points to avoid generating trigger
        signal twice.
        """
        trigger_signal = self.trigger_signal(record)
        trigger = signal_to_points(trigger_signal)
        return trigger, trigger_signal