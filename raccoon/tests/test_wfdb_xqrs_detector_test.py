import unittest

from os.path import dirname
import wfdb

from raccoon.detectors import WfdbXQRSDetector

RECORD_DIR = '/'.join([dirname(__file__), 'records'])
RECORD_NAME = '100'

class TestWfdbXQRSDetector(unittest.TestCase):

    def setUp(self):
        self.xqrs = WfdbXQRSDetector(name="MyXQRS")
        self.record = wfdb.rdrecord('/'.join([RECORD_DIR, RECORD_NAME]))

    def test_trigger_signal(self):
        """Does not return a trigger signal."""
        self.assertListEqual(self.xqrs.trigger_signal(self.record), [])

    def test_trigger(self):
        """All trigger should be between 0 and 1000 since the record length is
        1000 and hence no QRS complex can be before 0 or after 1000.
        """
        trigger = self.xqrs.trigger(self.record)
        self.assertIn(max(trigger), range(0, 1000))
        self.assertIn(min(trigger), range(0, 1000))

    def test_trigger_and_signal(self):
        """Tuple of trigger (points) and (trigger) signal expected."""
        trigger, signal = self.xqrs.trigger_and_signal(self.record)
        self.assertListEqual(signal, [])
        self.assertIn(max(trigger), range(0, 1000))
        self.assertIn(min(trigger), range(0, 1000))
