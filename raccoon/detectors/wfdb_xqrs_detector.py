import sys
sys.path.append("..")

from detectors.non_nn_detector import NonNNDetector

from wfdb import processing

class WfdbXQRSDetector(NonNNDetector):

    def __repr__(self):
        return "WFDB XQRS Detector"

    def __str__(self):
        return repr(self)

    # QRSDetector interface

    def trigger_signals(self, records):
        """No trigger signals generated here."""
        return [[] for record in records]

    def detect(self, records):
        return [
            processing.xqrs_detect(
                sig = record.p_signal.T[0],
                fs = record.fs,
                verbose = False)
            for record in records]

    def triggers_and_signals(self, records):
        return (self.trigger_signals(records), self.detect(records))