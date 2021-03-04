# disable Tensorflow logging
from os.path import dirname
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest

from qrsc.detectors import GarciaBerdonesDetector, WfdbXQRSDetector
from qrsc.utils import builders

class TestNameBuilder(unittest.TestCase):

    def setUp(self):
        self.name_builder = builders.NameBuilder()

    def test_already_in_use_initially_empty(self):
        self.assertListEqual(self.name_builder.already_in_use, [])

    def test_name_correctly_composed(self):
        name = self.name_builder.name()
        self.assertIn('_', name)
        adjective, animal = tuple(name.split('_'))
        self.assertIn(adjective, builders.ADJECTIVES)
        self.assertIn(animal, builders.ANIMALS)

    def test_name_added_to_already_in_use(self):
        name1 = self.name_builder.name()
        self.assertListEqual(self.name_builder.already_in_use, [name1])
        name2 = self.name_builder.name()
        self.assertListEqual(self.name_builder.already_in_use, [name1, name2])
        self.assertNotEqual(name1, name2)


class TestDetectorFactoryMethod(unittest.TestCase):

    def setUp(self):
        self.nb = builders.NameBuilder()

    def test_without_type(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.detector_from_dict({}, self.nb)
    
    def test_insufficient_arguments(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.detector_from_dict({"type": "XiangDetector"}, self.nb)

    def test_all_arguments(self):
        detector = builders.detector_from_dict({
            "type": "GarciaBerdonesDetector",
            "name": "TestGBD",
            "batch_size": 32,
            "window_size": 20,
            "epochs": 5,
            "gpus": 1}, self.nb)

        self.assertEqual(detector.__class__, GarciaBerdonesDetector)
        self.assertEqual(detector.name, "TestGBD")
        self.assertEqual(detector.batch_size, 32)
        self.assertEqual(detector.window_size, 20)
        self.assertEqual(detector.epochs, 5)
        self.assertEqual(detector.gpus, 1)

    def test_defaults(self):
        detector = builders.detector_from_dict({
            "type": "GarciaBerdonesDetector",
            "batch_size": 32,
            "window_size": 20}, self.nb)

        adjective, animal = detector.name.split('_')
        self.assertIn(adjective, builders.ADJECTIVES)
        self.assertIn(animal, builders.ANIMALS)
        self.assertEqual(detector.epochs, 1)


class TestEvaluatorFactoryMethod(unittest.TestCase):

    def test_input_dir_missing(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.evaluator_from_dict({"output_dir": "generated"})

    def test_output_dir_missing(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.evaluator_from_dict({"input_dir": "data"})

    def test_detectors_missing(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.evaluator_from_dict({
                "input_dir": "data", 
                "output_dir": "generated",
                "records": []})
    
    def test_records_missing(self):
        with self.assertRaises(builders.InsufficientConfiguration):
            builders.evaluator_from_dict({
                "input_dir": "data", 
                "output_dir": "generated",
                "detectors": [{"type": "WfdbXQRSDetector"}]})

    def test_basic_configuration(self):
        evaluator = builders.evaluator_from_dict({
                "input_dir": "data", 
                "output_dir": "generated",
                "detectors": [{"type": "WfdbXQRSDetector"}],
                "records": []})
        
        self.assertEqual(evaluator.input_dir, "data")
        self.assertEqual(evaluator.output_dir, "generated")
        self.assertEqual(evaluator.detectors[0].__class__, WfdbXQRSDetector)
        self.assertEqual(len(evaluator.records), 0)

    def test_name_builder_injection(self):
        name_builder = builders.NameBuilder()
        conf_dict = {
            "input_dir": "data",
            "output_dir": "generated",
            "detectors": [{"type": "WfdbXQRSDetector"}],
            "records": []}

        evaluator = builders.evaluator_from_dict(conf_dict, name_builder)

        self.assertEqual(
            name_builder.already_in_use[0],
            evaluator.detectors[0].name)

        self.assertEqual(len(name_builder.already_in_use), 1)
