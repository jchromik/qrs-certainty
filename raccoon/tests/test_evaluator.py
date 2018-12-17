from io import StringIO
from os import listdir, makedirs
from os.path import dirname, isabs, join, realpath, splitext
from shutil import rmtree
import json
import sys
import unittest

from raccoon.utils.builders import evaluator_from_dict

THIS_DIR = dirname(__file__)

def absify(path, dir):
    return path if isabs(path) else realpath(join(dir, path))


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        path = '/'.join([THIS_DIR, 'configurations', 'do_everything.json'])
        with open(path) as f:
            self.conf = json.load(f)
            self.conf["input_dir"] = absify(self.conf["input_dir"], dirname(f.name))
            self.conf["output_dir"] = absify(self.conf["output_dir"], dirname(f.name))
        self.evaluator = evaluator_from_dict(self.conf)

    def tearDown(self):
        rmtree(self.conf["output_dir"])

    def test_loocv(self):
        capture = StringIO()
        sys.stdout = capture
        self.evaluator.loocv()
        sys.stdout = sys.__stdout__

        file_names = listdir(self.conf["output_dir"])
        file_extensions = [splitext(name)[1] for name in file_names]
        self.assertIn('.svg', file_extensions)
        self.assertIn('.h5', file_extensions)
        self.assertIn('.atr', file_extensions)

    def test_kfold(self):
        capture = StringIO()
        sys.stdout = capture
        self.evaluator.kfold(k=2)
        sys.stdout = sys.__stdout__

        file_names = listdir(self.conf["output_dir"])
        file_extensions = [splitext(name)[1] for name in file_names]
        self.assertIn('.svg', file_extensions)
        self.assertIn('.h5', file_extensions)
        self.assertIn('.atr', file_extensions)

    def test_defined(self):
        capture = StringIO()
        sys.stdout = capture
        self.evaluator.defined(test_record_names=[self.conf["records"][0]])
        sys.stdout = sys.__stdout__

        file_names = listdir(self.conf["output_dir"])
        file_extensions = [splitext(name)[1] for name in file_names]
        self.assertIn('.svg', file_extensions)
        self.assertIn('.h5', file_extensions)
        self.assertIn('.atr', file_extensions)
