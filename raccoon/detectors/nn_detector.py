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
    def _trigger_signal(self, ecg_signal):
        """Use trained model to generate a trigger signal from an ECG signal."""
        pass

    # Common implementations

    def reset(self):
        """Rebuild model from scratch throwing away all weights."""
        self.model = self._build_model()

    def save_model(self, path):
        """Save trained model with weights to file."""
        self.model.save(path)

    def trigger_signals(self, ecg_signals):
        """Generate (multiple) trigger signals for (multiple) ECG signals."""
        return [self._trigger_signal(signal) for signal in ecg_signals]

    def detect(self, ecg_signals):
        """Find trigger points in (multiple) ECG signals."""
        return [
            signal_to_points(self._trigger_signal(signal))
            for signal in ecg_signals]

    def triggers_and_signals(self, ecg_signals):
        trigger_signals = self.trigger_signals(ecg_signals)
        return (
            trigger_signals,
            [signal_to_points(ts) for ts in trigger_signals])