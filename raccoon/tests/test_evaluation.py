from collections import OrderedDict
from io import StringIO
from os.path import dirname
import sys
import unittest

import wfdb

from raccoon.utils.annotationutils import trigger_points
from raccoon.evaluation import Evaluation
from raccoon.detectors import GarciaBerdonesDetector

THIS_DIR = dirname(__file__)
GENERATED_DIR = '/'.join([THIS_DIR, 'generated'])
RECORD_DIR = '/'.join([THIS_DIR, 'records'])
TRAIN_RECORD_NAMES = ['100', '101', '102']
TEST_RECORD_NAMES = ['103', '104', '105']

class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        detector = GarciaBerdonesDetector(
            name='MyGBD', batch_size=32, window_size=20)
        
        train_records = [
            wfdb.rdrecord('/'.join([RECORD_DIR, name]))
            for name in TRAIN_RECORD_NAMES]
        train_triggers = [
            trigger_points(
                wfdb.rdann('/'.join([RECORD_DIR, name]), extension='atr'))
            for name in TRAIN_RECORD_NAMES]
        
        test_records = [
             wfdb.rdrecord('/'.join([RECORD_DIR, name]))
            for name in TEST_RECORD_NAMES]
        test_triggers = [
            trigger_points(
                wfdb.rdann('/'.join([RECORD_DIR, name]), extension='atr'))
            for name in TEST_RECORD_NAMES]
        
        cls.evaluator = Evaluation(
            output_dir=GENERATED_DIR,
            evaluation_id=0,
            detector=detector,
            train_records=train_records,
            train_triggers=train_triggers,
            test_records=test_records,
            test_triggers=test_triggers,
            trigger_distance=5)

        capture = StringIO()
        sys.stdout = capture
        cls.evaluator.run()
        sys.stdout = sys.__stdout__

    def test_report(self):
        report = self.evaluator.report()
        self.assertEqual(len(report), len(TEST_RECORD_NAMES))
        for entry in report:
            self.assertIsInstance(entry, OrderedDict)