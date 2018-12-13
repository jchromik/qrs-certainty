from . import NonNNDetector

from wfdb import processing

class WfdbGQRSDetector(NonNNDetector):

    def __init__(self, name):
        self.name = name

    # QRSDetector interface

    def trigger_signal(self, record):
        """No trigger signals generated here."""
        return []

    def trigger(self, record):
        return processing.gqrs_detect(
            sig = record.p_signal.T[0],
            fs = record.fs)
    
    def trigger_and_signal(self, record):
        return self.trigger(record), self.trigger_signal(record)