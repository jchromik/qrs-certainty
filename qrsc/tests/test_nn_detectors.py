from os import makedirs
from os.path import dirname, exists
from shutil import rmtree
import sys
import unittest

from io import StringIO
from keras.models import Sequential, Model
import wfdb

from qrsc.detectors import (
    GarciaBerdonesDetector, RaccoonDetector, SarlijaDetector, XiangDetector)
from qrsc.utils.annotationutils import trigger_points

THIS_DIR = dirname(__file__)
GENERATED_DIR = '/'.join([THIS_DIR, 'generated'])
RECORD_DIR = '/'.join([THIS_DIR, 'records'])
RECORD_NAMES = ['100', '101', '102']


class TestNNDetectors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        makedirs(GENERATED_DIR, exist_ok=True)

    @classmethod
    def tearDown(cls):
        rmtree(GENERATED_DIR, ignore_errors=True)

    def setUp(self):
        self.garcia = GarciaBerdonesDetector(
            name="MyGarcia", batch_size=32, window_size=20)
        self.raccoon = RaccoonDetector(
            name="MyRaccoon", batch_size=32, window_size=40, detection_size=10,
            winavg_sizes=[1, 3])
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
        self.assertIn("Threshold: 0.8", splitstring[3])
        self.assertIn("Tolerance: 10", splitstring[4])
        self.assertIn("Training Epochs: 1", splitstring[5])
        self.assertIn("Number of GPUs used: 0", splitstring[6])

    def test_str_raccoon(self):
        splitstring = str(self.raccoon).splitlines()
        self.assertIn("MyRaccoon (RaccoonDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 40", splitstring[2])
        self.assertIn("Detection Size: 10", splitstring[3])
        self.assertIn("Window Average Sizes: [1, 3]", splitstring[4])
        self.assertIn("Threshold: 0.8", splitstring[5])
        self.assertIn("Tolerance: 10", splitstring[6])
        self.assertIn("Training Epochs: 1", splitstring[7])
        self.assertIn("Number of GPUs used: 0", splitstring[8])

    def test_str_sarlija(self):
        splitstring = str(self.sarlija).splitlines()
        self.assertIn("MySarlija (SarlijaDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 20", splitstring[2])
        self.assertIn("Detection Size: 10", splitstring[3])
        self.assertIn("Threshold: 0.8", splitstring[4])
        self.assertIn("Tolerance: 10", splitstring[5])
        self.assertIn("Training Epochs: 1", splitstring[6])
        self.assertIn("Number of GPUs used: 0", splitstring[7])

    def test_str_xiang(self):
        splitstring = str(self.xiang).splitlines()
        self.assertIn("MyXiang (XiangDetector)", splitstring[0])
        self.assertIn("Batch Size: 32", splitstring[1])
        self.assertIn("Window Size: 40", splitstring[2])
        self.assertIn("Detection Size: 10", splitstring[3])
        self.assertIn("Aux Ratio: 5", splitstring[4])
        self.assertIn("Threshold: 0.8", splitstring[5])
        self.assertIn("Tolerance: 10", splitstring[6])
        self.assertIn("Training Epochs: 1", splitstring[7])
        self.assertIn("Number of GPUs used: 0", splitstring[8])

    def test_train_and_predict(self):
        """First train, than generate trigger signal.
        Trigger signal should be approximately 1000 samples long and contain
        values between 0 and 1.
        """
        capture = StringIO()
        sys.stdout = capture
        
        for detector in [self.garcia, self.raccoon, self.sarlija, self.xiang]:
            detector.train(self.records, self.triggers)
            trigger, signal = detector.trigger_and_signal(self.records[0])

            self.assertAlmostEqual(len(signal), 1000, places=-2)
            self.assertTrue(0 <= max(signal) <= 1)
            self.assertTrue(0 <= min(signal) <= 1)

            self.assertTrue(all(map(lambda t: t in range(0, 1000), trigger)))

            trigger = detector.trigger(self.records[0])
            self.assertTrue(all(map(lambda t: t in range(0, 1000), trigger)))

        sys.stdout = sys.__stdout__

    def test_build_model(self):
        self.assertIsInstance(self.garcia._build_model(), Sequential)
        self.assertIsInstance(self.raccoon._build_model(), Model)
        self.assertIsInstance(self.sarlija._build_model(), Sequential)
        self.assertIsInstance(self.xiang._build_model(), Model)

    def test_reset(self):
        garcia_original = self.garcia.model
        raccoon_original = self.raccoon.model
        sarlija_original = self.sarlija.model
        xiang_original = self.xiang.model
        self.garcia.reset()
        self.raccoon.reset()
        self.sarlija.reset()
        self.xiang.reset()
        self.assertNotEqual(self.garcia.model, garcia_original)
        self.assertNotEqual(self.raccoon.model, raccoon_original)
        self.assertNotEqual(self.sarlija.model, sarlija_original)
        self.assertNotEqual(self.xiang.model, xiang_original)

    def test_save_model(self):
        garcia_path = '/'.join([GENERATED_DIR, 'garcia.h5'])
        self.garcia.save_model(garcia_path)
        self.assertTrue(exists(garcia_path))

        raccoon_path = '/'.join([GENERATED_DIR, 'raccoon.h5'])
        self.raccoon.save_model(raccoon_path)
        self.assertTrue(exists(raccoon_path))

        sarlija_path = '/'.join([GENERATED_DIR, 'sarlija.h5'])
        self.sarlija.save_model(sarlija_path)
        self.assertTrue(exists(sarlija_path))

        xiang_path = '/'.join([GENERATED_DIR, 'xiang.h5'])
        self.xiang.save_model(xiang_path)
        self.assertTrue(exists(xiang_path))