from abc import abstractmethod
from os.path import dirname
from os import makedirs

from . import QRSDetector
from ..utils.triggerutils import signal_to_points


class NNDetector(QRSDetector):

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
        return signal_to_points(self.trigger_signal(record))

    def trigger_and_signal(self, record):
        """Return trigger signal and trigger points to avoid generating trigger
        signal twice.
        """
        trigger_signal = self.trigger_signal(record)
        trigger = signal_to_points(trigger_signal)
        return trigger, trigger_signal