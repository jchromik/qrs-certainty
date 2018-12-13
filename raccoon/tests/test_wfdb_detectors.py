import unittest

from os.path import dirname
import wfdb

from raccoon.detectors import WfdbXQRSDetector, WfdbGQRSDetector

RECORD_DIR = '/'.join([dirname(__file__), 'records'])
RECORD_NAME = '100'

class TestWfdbDetectors(unittest.TestCase):

    def setUp(self):
        self.gqrs = WfdbGQRSDetector(name="MyGQRS")
        self.xqrs = WfdbXQRSDetector(name="MyXQRS")
        self.record = wfdb.rdrecord('/'.join([RECORD_DIR, RECORD_NAME]))

    def test_trigger_signal(self):
        """Does not return a trigger signal."""
        self.assertListEqual(self.gqrs.trigger_signal(self.record), [])
        self.assertListEqual(self.xqrs.trigger_signal(self.record), [])

    def test_trigger(self):
        """All trigger should be between 0 and 1000 since the record length is
        1000 and hence no QRS complex can be before 0 or after 1000.
        """
        gqrs_trigger = self.gqrs.trigger(self.record)
        xqrs_trigger = self.xqrs.trigger(self.record)

        self.assertIn(max(gqrs_trigger), range(0, 1000))
        self.assertIn(min(gqrs_trigger), range(0, 1000))

        self.assertIn(max(xqrs_trigger), range(0, 1000))
        self.assertIn(min(xqrs_trigger), range(0, 1000))

    def test_trigger_and_signal(self):
        """Tuple of trigger (points) and (trigger) signal expected."""
        gqrs_trigger, gqrs_signal = self.gqrs.trigger_and_signal(self.record)
        xqrs_trigger, xqrs_signal = self.xqrs.trigger_and_signal(self.record)

        self.assertListEqual(gqrs_signal, [])
        self.assertIn(max(gqrs_trigger), range(0, 1000))
        self.assertIn(min(gqrs_trigger), range(0, 1000))

        self.assertListEqual(xqrs_signal, [])
        self.assertIn(max(xqrs_trigger), range(0, 1000))
        self.assertIn(min(xqrs_trigger), range(0, 1000))
