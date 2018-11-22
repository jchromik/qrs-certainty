from abc import ABC, abstractmethod

class QRSDetector(ABC):

    @abstractmethod
    def train(self, ecg_signals, trigger_points):
        """Use input as training data to train model."""
        pass

    @abstractmethod
    def trigger_signals(self, ecg_signals):
        """Generate a trigger signal as intermediate representation.""" 
        pass

    @abstractmethod
    def detect(self, ecg_signals):
        """Return a list of positions where QRS complexes are detected."""
        pass

    @abstractmethod
    def triggers_and_signals(self, ecg_signals):
        """Returns list of QRS positions and trigger signal with one prediction
        step. Like trigger_signal and detect combined. More efficient than
        calling trigger_signals and detect independently.
        """
        pass

    @abstractmethod
    def reset(self):
        """Make the model untrained again."""
        pass

    @abstractmethod
    def save_model(self, path):
        """Save trained model (if any) to a file."""
        pass