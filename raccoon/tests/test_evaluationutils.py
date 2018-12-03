import sys
sys.path.append("..")

import math
import unittest

from raccoon.utils import evaluationutils as eu

class TestEvaluationUtils(unittest.TestCase):

    def test_triggers_metrics(self):
        true = [
            [10, 55, 80, 100],
            [25, 70, 92],
            [11, 36, 49, 78, 99]
        ]
        pred = [
            [10, 11, 85, 95, 106],
            [11, 26, 65],
            [11, 30, 33, 36, 40]
        ]
        tolerance = 5
        self.assertEqual(
            eu.triggers_metrics(true, pred, tolerance),
            (7, 0, 6, 5))

    def test_sensitivity(self):
        self.assertEqual(eu.sensitivity(3, 7), 0.3)
        self.assertTrue(math.isnan(eu.sensitivity(0, 0)))

    def test_ppv(self):
        self.assertEqual(eu.ppv(3, 7), 0.3)
        self.assertTrue(math.isnan(eu.ppv(0, 0)))

    def test_f1(self):
        self.assertEqual(eu.f1(3, 7, 7), 0.3)
        self.assertTrue(math.isnan(eu.f1(0, 0, 0)))
