from collections import OrderedDict
from io import StringIO
from os import makedirs
from os.path import dirname, exists
from shutil import rmtree
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
        makedirs(GENERATED_DIR, exist_ok=True)

        cls.detector = GarciaBerdonesDetector(
            name='MyGBD', batch_size=32, window_size=20)
        
        cls.train_records = [
            wfdb.rdrecord('/'.join([RECORD_DIR, name]))
            for name in TRAIN_RECORD_NAMES]
        cls.train_triggers = [
            trigger_points(
                wfdb.rdann('/'.join([RECORD_DIR, name]), extension='atr'))
            for name in TRAIN_RECORD_NAMES]
        
        cls.test_records = [
             wfdb.rdrecord('/'.join([RECORD_DIR, name]))
            for name in TEST_RECORD_NAMES]
        cls.test_triggers = [
            trigger_points(
                wfdb.rdann('/'.join([RECORD_DIR, name]), extension='atr'))
            for name in TEST_RECORD_NAMES]
        
        cls.evaluation = Evaluation(
            output_dir=GENERATED_DIR,
            evaluation_id=0,
            detector=cls.detector,
            train_records=cls.train_records,
            train_triggers=cls.train_triggers,
            test_records=cls.test_records,
            test_triggers=cls.test_triggers,
            trigger_distance=5)

        capture = StringIO()
        sys.stdout = capture
        cls.evaluation.run()
        sys.stdout = sys.__stdout__

    @classmethod
    def tearDownClass(cls):
        rmtree(GENERATED_DIR)

    def test_report(self):
        report = self.evaluation.report()
        self.assertEqual(len(report), len(TEST_RECORD_NAMES))
        for entry in report:
            self.assertIsInstance(entry, OrderedDict)

    def test_save_annotations(self):
        records = self.test_records
        triggers = self.evaluation.detected_triggers

        self.evaluation.save_annotations()
        
        for record, trigger in zip(records, triggers):
            if not trigger: continue # no trigger found --> no file to save
            file_name = self.evaluation._file_name_for(record)
            self.assertTrue(exists('{}/{}.atr'.format(GENERATED_DIR, file_name)))

    def test_save_model(self):
        self.evaluation.save_model()
        file_name = self.evaluation._file_name()
        self.assertTrue(exists('{}/{}.h5'.format(GENERATED_DIR, file_name)))

    def test_plot_detections(self):
        self.evaluation.plot_detections(xlim=1000)
        for record in self.test_records:
            file_name = self.evaluation._file_name_for(record)
            self.assertTrue(exists('{}/{}.svg'.format(GENERATED_DIR, file_name)))
