import sys
sys.path.append("..")

import unittest

from evaluationutils import points_to_metrics

class TestEvaluationUtils(unittest.TestCase):

    def test_points_to_metrics(self):
        true = [10, 55, 80, 100]
        pred = [10, 11, 85, 95, 106]
        tolerance = 5
        self.assertEqual(
            points_to_metrics(true, pred, tolerance),
            (3, 0, 2, 1))
            
if __name__ == '__main__':
    unittest.main()