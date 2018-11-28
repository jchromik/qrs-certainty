from abc import ABC, abstractmethod

class QRSDetector(ABC):

    def __repr__(self):
        return "{} ({})".format(self.name, self.__class__.__name__)

    @abstractmethod
    def train(self, records, triggers):
        """Use input as training data to train model."""
        pass

    @abstractmethod
    def trigger_signals(self, records):
        """Generate a trigger signal as intermediate representation.""" 
        pass

    @abstractmethod
    def detect(self, records):
        """Return a list of positions where QRS complexes are detected."""
        pass

    @abstractmethod
    def triggers_and_signals(self, records):
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