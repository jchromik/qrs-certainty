from . import NonNNDetector

from wfdb import processing

class WfdbXQRSDetector(NonNNDetector):

    def __init__(self, name):
        self.name = name

    # QRSDetector interface

    def trigger_signal(self, record):
        """No trigger signals generated here."""
        return []

    def trigger(self, record):
        return processing.xqrs_detect(
            sig = record.p_signal.T[0],
            fs = record.fs,
            verbose = False)

    def trigger_and_signal(self, record):
        return self.trigger_signal(record), self.trigger(record)