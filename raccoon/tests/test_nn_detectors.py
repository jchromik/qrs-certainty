import unittest

from os.path import dirname
import wfdb

from raccoon.detectors import (
    GarciaBerdonesDetector, SarlijaDetector, XiangDetector)
from raccoon.utils.annotationutils import trigger_points

THIS_DIR = dirname(__file__)
RECORD_DIR = '/'.join([THIS_DIR, 'records'])
RECORD_NAMES = ['100', '101', '102']


class TestNNDetectors(unittest.TestCase):

    def setUp(self):
        self.garcia = GarciaBerdonesDetector(
            name="MyGarcia", batch_size=32, window_size=20)
        self.sarlija = SarlijaDetector(
            name="MySarlija", batch_size=32, window_size=20, detection_size=10)
        self.xiang = XiangDetector(
            name="MyXiang", batch_size=32, window_size=40, detection_size=10,
            aux_ratio=5)

        self.records = [
            wfdb.rdrecord('/'.join([RECORD_DIR, record_name]))
            for record_name in RECORD_NAMES]
        
        self.triggers = [
            trigger_points(
                wfdb.rdann(
                    '/'.join([RECORD_DIR, record_name]),
                    extension='atr'))
            for record_name in RECORD_NAMES]
    
    def test_str_garcia(self):
        splitstring = str(self.garcia).splitlines()
        self.assertIn("MyGarcia (GarciaBerdonesDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 20", splitstring[2])
        self.assertIn("Training Epochs: 1", splitstring[3])
        self.assertIn("Number of GPUs used: 0", splitstring[4])

    def test_str_sarlija(self):
        splitstring = str(self.sarlija).splitlines()
        self.assertIn("MySarlija (SarlijaDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 20", splitstring[2])
        self.assertIn("Detection Size: 10", splitstring[3])
        self.assertIn("Training Epochs: 1", splitstring[4])
        self.assertIn("Number of GPUs used: 0", splitstring[5])

    def test_str_xiang(self):
        splitstring = str(self.xiang).splitlines()
        self.assertIn("MyXiang (XiangDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 40", splitstring[2])
        self.assertIn("Detection Size: 10", splitstring[3])
        self.assertIn("Aux Ratio: 5", splitstring[4])
        self.assertIn("Training Epochs: 1", splitstring[5])
        self.assertIn("Number of GPUs used: 0", splitstring[6])

    def test_train_and_trigger_signal(self):
        """First train, than generate trigger signal.
        Trigger signal should be approximately 1000 samples long and contain
        values between 0 and 1.
        """
        for detector in [self.garcia, self.sarlija, self.xiang]:
            detector.train(self.records, self.triggers)
            signal = detector.trigger_signal(self.records[0])
            self.assertAlmostEqual(len(signal), 1000, places=-2)
            self.assertTrue(0 <= max(signal) <= 1)
            self.assertTrue(0 <= min(signal) <= 1)
