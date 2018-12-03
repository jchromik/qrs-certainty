from abc import ABC, abstractmethod

class QRSDetector(ABC):

    def __repr__(self):
        return f"{self.name} ({self.__class__.__name__})"

    def trigger_signals(self, records):
        """Generate (multiple) trigger signals for multiple ECG recordings."""
        return [self.trigger_signal(record) for record in records]

    def detect(self, records):
        """Find trigger points in multiple ECG recordings."""
        return [self.trigger(record) for record in records]

    def triggers_and_signals(self, records):
        """Convenience wrapper of trigger_and_signal for multiple ECG
        recordings.
        """
        triggers, signals = tuple(zip(*[
            self.trigger_and_signal(record) for record in records]))
        return list(signals), list(triggers)

    @abstractmethod
    def train(self, records, triggers):
        """Use input as training data to train model."""
        pass

    @abstractmethod
    def trigger_signal(self, records):
        """Generate a trigger signal as intermediate representation.""" 
        pass

    @abstractmethod
    def trigger(self, records):
        """Return a list of positions where QRS complexes are detected."""
        pass

    @abstractmethod
    def trigger_and_signal(self, records):
        """Returns list of QRS positions (trigger) and trigger signal with one
        prediction step.
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