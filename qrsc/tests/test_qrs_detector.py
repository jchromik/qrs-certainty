from os.path import dirname
import unittest

import wfdb

from qrsc.detectors import WfdbXQRSDetector

THIS_DIR = dirname(__file__)
RECORD_DIR = '/'.join([THIS_DIR, 'records'])
RECORD_NAMES = ['100', '101', '102']


class TestQRSDetector(unittest.TestCase):

    """Test functions implemented in QRSDetector ABC using one QRS detector
    (i.e. WfdbXQRSDetector) as representative for all."""

    def setUp(self):
        self.records = [
            wfdb.rdrecord('/'.join([RECORD_DIR, record_name]))
            for record_name in RECORD_NAMES]

        self.xqrs = WfdbXQRSDetector(name="MyXQRS")

    def test_trigger_signals(self):
        signals = self.xqrs.trigger_signals(self.records)
        self.assertEqual(len(signals), len(RECORD_NAMES))
        # WfdbXQRSDetector produces empty trigger signals:
        self.assertTrue(all(map(lambda s: s == [], signals)))

    def test_detect(self):
        triggers = self.xqrs.detect(self.records)
        self.assertEqual(len(triggers), len(RECORD_NAMES))
        for trigger in triggers:
            self.assertTrue(all(map(lambda t: t in range(0, 1000), trigger)))

    def test_triggers_and_signals(self):
        triggers, signals = self.xqrs.triggers_and_signals(self.records)
        self.assertEqual(len(signals), len(RECORD_NAMES))
        self.assertEqual(len(triggers), len(RECORD_NAMES))
        # WfdbXQRSDetector produces empty trigger signals:
        self.assertTrue(all(map(lambda s: s == [], signals)))
        for trigger in triggers:
            self.assertTrue(all(map(lambda t: t in range(0, 1000), trigger)))