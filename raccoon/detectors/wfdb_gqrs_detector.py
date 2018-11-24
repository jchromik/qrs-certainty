import sys
sys.path.append("..")

from detectors.non_nn_detector import NonNNDetector

from wfdb import processing

class WfdbGQRSDetector(NonNNDetector):

    def __repr__(self):
        return "WFDB GQRS Detector"

    def __str__(self):
        return repr(self)

    # QRSDetector interface

    def trigger_signals(self, records):
        """No trigger signals generated here."""
        return [[] for record in records]

    def detect(self, records):
        return [
            processing.gqrs_detect(
                sig = record.p_signal.T[0],
                fs = record.fs)
            for record in records]
    
    def triggers_and_signals(self, records):
        return (self.trigger_signals(records), self.detect(records))