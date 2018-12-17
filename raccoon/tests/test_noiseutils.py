from random import randrange
import unittest

import numpy as np

import raccoon.utils.noiseutils as nu

class TestNoiseUtils(unittest.TestCase):

    def test_power(self):
        self.assertEqual(nu.power([-1, 3, 2, -2]), 4.5)

    def test_snr(self):
        self.assertEqual(nu.snr(
            signal=[1, 2, 3, 2, 1], # power = 3.8
            noise=[1, 2, 1]), # power = 2
            1.9) # snr = power(signal) / power(noise) = 3.8/2 = 1.9

    def test_scale(self):
        self.assertListEqual(nu.scale([1, 2, 3, 4, 5], 2), [2, 4, 6, 8, 10])

    def test_repeat_until(self):
        self.assertListEqual(nu.repeat_until([1,2,3], 7), [1, 2, 3, 1, 2, 3, 1])

    def test_add_shortest(self):
        self.assertListEqual(
            nu.add_shortest([1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]),
            [14, 17, 20])

    def test_add_repeating(self):
        self.assertListEqual(
            nu.add_repeating([1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]),
            [14, 17, 20, 20])

    def test_apply_noise(self):
        for _ in range(100):
            signal = [randrange(100) for _ in range(100)]
            noise = [randrange(100) for _ in range(20)]
            target_snr = randrange(10) + 1
            noisy_signal = nu.apply_noise(signal, noise, target_snr)
            added_noise = list(np.subtract(noisy_signal, signal))
            self.assertAlmostEqual(nu.snr(signal, added_noise), target_snr)