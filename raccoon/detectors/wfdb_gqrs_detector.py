import sys
sys.path.append("..")

from detectors.non_nn_detector import NonNNDetector

from wfdb import processing

class WfdbGQRSDetector(NonNNDetector):

    # Initializaion

    def __init__(self, signal_freq):
        self.signal_freq = signal_freq

    def __repr__(self):
        return "WFDB GQRS Detector"

    def __str__(self):
        return repr(self)

    # QRSDetector interface

    def trigger_signals(self, ecg_signals):
        """No trigger signals generated here."""
        return [[] for signal in ecg_signals]

    def detect(self, ecg_signals):
        return [
            processing.gqrs_detect(
                sig = signal,
                fs = self.signal_freq)
            for signal in ecg_signals]
    
    def triggers_and_signals(self, ecg_signals):
        return (self.trigger_signals(ecg_signals), self.detect(ecg_signals))