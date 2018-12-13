import unittest

from os.path import dirname
import wfdb

from raccoon.detectors import PanTompkinsDetector

THIS_DIR = dirname(__file__)
RECORD_DIR = '/'.join([THIS_DIR, 'records'])
RECORD_NAME = '100'


class TestPanTompkinsDetectors(unittest.TestCase):

    def setUp(self):
        self.ptd = PanTompkinsDetector("MyPTD", window_size=10)
        self.record = wfdb.rdrecord('/'.join([RECORD_DIR, RECORD_NAME]))

    def test_str(self):
        self.assertIn("MyPTD (PanTompkinsDetector)",
                      str(self.ptd).splitlines()[0])
        self.assertIn("Moving Window Size: 10", str(self.ptd).splitlines()[1])

    def test_trigger_signal(self):
        """Trigger signal should be approximately 1000 samples long and contain
        values between 0 and 1.
        """
        signal = self.ptd.trigger_signal(self.record)
        self.assertAlmostEqual(len(signal), 1000, places=-1)
        self.assertTrue(0 <= max(signal) <= 1)
        self.assertTrue(0 <= min(signal) <= 1)

    def test_trigger(self):
        """Trigger points should be found between 0 and 1000."""
        trigger = self.ptd.trigger(self.record)
        self.assertTrue(0 <= max(trigger) <= 1000)
        self.assertTrue(0 <= min(trigger) <= 1000)

    def test_trigger_and_signal(self):
        """Should return tuple of trigger (points) and (trigger) signal."""
        trigger, signal = self.ptd.trigger_and_signal(self.record)

        self.assertAlmostEqual(len(signal), 1000, places=-1)
        self.assertTrue(0 <= max(signal) <= 1)
        self.assertTrue(0 <= min(signal) <= 1)

        self.assertTrue(0 <= max(trigger) <= 1000)
        self.assertTrue(0 <= min(trigger) <= 1000)
