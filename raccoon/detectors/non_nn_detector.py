import sys
sys.path.append("..")

from detectors.qrs_detector import QRSDetector

class NonNNDetector(QRSDetector):

    def train(self, ecg_signals, trigger_points):
        """Do nothing since this is not machine learning."""
        pass

    def reset(self):
        """Do nothing since no training data have been used."""
        pass

    def save_model(self, path):
        """Do nothing, since there is no model."""
        pass