import numpy as np
import numpy.testing as npt
import unittest

from raccoon.detectors import WindowGenerator
from raccoon.utils.triggerutils import points_to_signal

SIGNALS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

TRIGGERS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]

class TestWindowGenerator(unittest.TestCase):

    def test_len(self):
        gen = WindowGenerator(SIGNALS, 8, 4)
        self.assertEqual(len(gen), 2)

    def test_get_item_without_triggers(self):
        gen = WindowGenerator(SIGNALS, 2, 4)
        npt.assert_allclose(gen[0], [[0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6]])
        npt.assert_allclose(gen[1], [[0.1, 0.2, 0.6, 0.0], [0.2, 0.6, 0.0, 0.3]])
        npt.assert_allclose(gen[3], [[0.5, 0.0, 0.2, 0.5], [0.0, 0.2, 0.5, 0.8]])

    def test_get_item_with_triggers(self):
        gen = WindowGenerator(SIGNALS, 2, 4, TRIGGERS)
        trigger_signal = points_to_signal(TRIGGERS[0], len(SIGNALS[0]), 6)
        npt.assert_allclose(gen[0][0], [[0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6]])
        npt.assert_allclose(gen[0][1], [trigger_signal[2], trigger_signal[3]])