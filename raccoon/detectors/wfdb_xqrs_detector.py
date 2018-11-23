import sys
sys.path.append("..")

from detectors.non_nn_detector import NonNNDetector

from wfdb import processing

class WfdbXQRSDetector(NonNNDetector):

    # Initializaion

    def __init__(self, signal_freq):
        self.signal_freq = signal_freq

    def __repr__(self):
        return "WFDB XQRS Detector"

    def __str__(self):
        return "\n".join([
            repr(self),
            "\tSignal Frequency: {}".format(self.signal_freq)])

    # QRSDetector interface

    def trigger_signals(self, ecg_signals):
        """No trigger signals generated here."""
        return [[] for signal in ecg_signals]

    def detect(self, ecg_signals):
        return [
            processing.xqrs_detect(
                sig = signal,
                fs = self.signal_freq,
                verbose = False)
            for signal in ecg_signals]

    def triggers_and_signals(self, ecg_signals):
        return (self.trigger_signals(ecg_signals), self.detect(ecg_signals))