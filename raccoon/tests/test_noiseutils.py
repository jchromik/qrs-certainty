from itertools import product
from os.path import dirname
from random import randrange
import unittest

import numpy as np
import numpy.testing as npt
import wfdb

import raccoon.utils.noiseutils as nu

THIS_DIR = dirname(__file__)
RECORD_DIR = '/'.join([THIS_DIR, 'records'])


class TestNoiseUtils(unittest.TestCase):

    def test_power(self):
        self.assertEqual(nu.power([-1, 3, 2, -2]), 4.5)

    def test_snr(self):
        self.assertEqual(nu.snr(
            signal=[1, 2, 3, 2, 1],  # power = 3.8
            noise=[1, 2, 1]),  # power = 2
            1.9)  # snr = power(signal) / power(noise) = 3.8/2 = 1.9

    def test_snr_to_snrdb(self):
        self.assertEqual(nu.snr_to_snrdb(100), 20)

    def test_snrdb_to_snr(self):
        self.assertEqual(nu.snrdb_to_snr(20), 100)

    def test_scale(self):
        npt.assert_array_equal(nu.scale([1, 2, 3, 4, 5], 2), [2, 4, 6, 8, 10])

    def test_repeat_until(self):
        npt.assert_array_equal(
            nu.repeat_until([1, 2, 3], 7),
            [1, 2, 3, 1, 2, 3, 1])

    def test_add_repeating(self):
        npt.assert_array_equal(
            nu.add_repeating([1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11]),
            [14, 17, 20, 20])

    def test_apply_noise_both_iterable(self):
        """Applying noise when signal and noise are both iterable."""
        for _ in range(100):
            signal = [randrange(100) for _ in range(100)]
            noise = [randrange(100) for _ in range(20)]
            target_snr = randrange(10) + 1
            contaminated_signal = nu.apply_noise(signal, noise, target_snr)
            added_noise = list(np.subtract(contaminated_signal, signal))
            self.assertAlmostEqual(nu.snr(signal, added_noise), target_snr)

    def test_apply_noise_signal_record(self):
        """Applying noise when signal is WFDB Record and noise is Iterable."""

        signal_names = ['100', '101', '102', '103', '104', '105']
        noise_names = [
            'em500', 'em1000', 'em2000',
            'ma500', 'ma1000', 'ma2000',
            'bw500', 'bw1000', 'bw2000']

        for signal_name, noise_name in product(signal_names, noise_names):
            signal_record = wfdb.rdrecord('/'.join([RECORD_DIR, signal_name]))
            noise_record = wfdb.rdrecord('/'.join([RECORD_DIR, noise_name]))

            signal = signal_record.p_signal.T[0]
            noise = noise_record.p_signal.T[0]

            target_snr = randrange(10) + 1

            contaminated_record = nu.apply_noise(
                signal_record, noise, target_snr)
            contaminated_signal = contaminated_record.p_signal.T[0]

            added_noise = list(np.subtract(contaminated_signal, signal))
            self.assertAlmostEqual(nu.snr(signal, added_noise), target_snr)

    def test_apply_noise_noise_record(self):
        """Applying noise when signal is Iterable and noise is WFDB Record."""

        signal_names = ['100', '101', '102', '103', '104', '105']
        noise_names = [
            'em500', 'em1000', 'em2000',
            'ma500', 'ma1000', 'ma2000',
            'bw500', 'bw1000', 'bw2000']

        for signal_name, noise_name in product(signal_names, noise_names):
            signal_record = wfdb.rdrecord('/'.join([RECORD_DIR, signal_name]))
            noise_record = wfdb.rdrecord('/'.join([RECORD_DIR, noise_name]))

            signal = signal_record.p_signal.T[0]
            target_snr = randrange(10) + 1

            contaminated_signal = nu.apply_noise(
                signal, noise_record, target_snr)

            added_noise = list(np.subtract(contaminated_signal, signal))
            self.assertAlmostEqual(nu.snr(signal, added_noise), target_snr)

    def test_apply_noise_both_record(self):
        """Applying noise when signal and noise are both WFDB Records."""

        signal_names = ['100', '101', '102', '103', '104', '105']
        noise_names = [
            'em500', 'em1000', 'em2000',
            'ma500', 'ma1000', 'ma2000',
            'bw500', 'bw1000', 'bw2000']

        for signal_name, noise_name in product(signal_names, noise_names):
            signal_record = wfdb.rdrecord('/'.join([RECORD_DIR, signal_name]))
            noise_record = wfdb.rdrecord('/'.join([RECORD_DIR, noise_name]))

            signal = signal_record.p_signal.T[0]
            target_snr = randrange(10) + 1

            contaminated_record = nu.apply_noise(
                signal_record, noise_record, target_snr)
            contaminated_signal = contaminated_record.p_signal.T[0]

            added_noise = list(np.subtract(contaminated_signal, signal))
            self.assertAlmostEqual(nu.snr(signal, added_noise), target_snr)

    def test_apply_noise_db(self):
        """Appling noise works, when SNR is given in decibel (dB)."""

        signal_names = ['100', '101', '102', '103', '104', '105']
        noise_names = [
            'em500', 'em1000', 'em2000',
            'ma500', 'ma1000', 'ma2000',
            'bw500', 'bw1000', 'bw2000']

        for signal_name, noise_name in product(signal_names, noise_names):
            signal_record = wfdb.rdrecord('/'.join([RECORD_DIR, signal_name]))
            noise_record = wfdb.rdrecord('/'.join([RECORD_DIR, noise_name]))

            signal = signal_record.p_signal.T[0]
            target_snr_db = randrange(10) + 1

            contaminated_record = nu.apply_noise_db(
                signal_record, noise_record, target_snr_db)
            contaminated_signal = contaminated_record.p_signal.T[0]

            added_noise = list(np.subtract(contaminated_signal, signal))
            self.assertAlmostEqual(
                nu.snrdb(signal, added_noise), target_snr_db)
