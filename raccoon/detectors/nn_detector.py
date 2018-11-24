import sys
sys.path.append("..")

from detectors.qrs_detector import QRSDetector
from triggerutils import signal_to_points

from abc import abstractmethod

class NNDetector(QRSDetector):

    # Additional abstract method

    @abstractmethod
    def _build_model(self):
        """Build the detector-specific neural network (model)."""
        pass

    @abstractmethod
    def _trigger_signal(self, records):
        """Use trained model to generate a trigger signal from an ECG recording."""
        pass

    # Common implementations

    def reset(self):
        """Rebuild model from scratch throwing away all weights."""
        self.model = self._build_model()

    def save_model(self, path):
        """Save trained model with weights to file."""
        self.model.save(path)

    def trigger_signals(self, records):
        """Generate (multiple) trigger signals for (multiple) ECG recordings."""
        return [self._trigger_signal(record) for record in records]

    def detect(self, records):
        """Find trigger points in (multiple) ECG recordings."""
        return [
            signal_to_points(self._trigger_signal(record))
            for record in records]

    def triggers_and_signals(self, records):
        trigger_signals = self.trigger_signals(records)
        return (
            trigger_signals,
            [signal_to_points(ts) for ts in trigger_signals])