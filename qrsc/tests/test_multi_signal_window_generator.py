import unittest

import numpy.testing as npt

from qrsc.generators import MultiSignalWindowGenerator

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

AUX_CHUNKS = [
    [0.4, 0.2, 0.0, 0.5, 0.4],
    [0.0, 0.5, 0.4, 0.4, 0.0, 0.2],
    [0.2, 0.9, 0.9, 0.7, 0.1]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]


class TestMultiSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.mswg_train = MultiSignalWindowGenerator(
            signals=[SIGNAL_CHUNKS, AUX_CHUNKS],
            batch_size=2,
            window_sizes=[4, 2],
            trigger_chunks=TRIGGER_CHUNKS,
            detection_size=1)

        self.mswg_train_wrap = MultiSignalWindowGenerator(
            signals=[SIGNAL_CHUNKS, AUX_CHUNKS],
            batch_size=2,
            window_sizes=[4, 2],
            trigger_chunks=TRIGGER_CHUNKS,
            detection_size=1,
            wrap_samples=True)

        self.mswg_test = MultiSignalWindowGenerator(
            signals=[SIGNAL_CHUNKS, AUX_CHUNKS],
            batch_size=2,
            window_sizes=[4, 2])

        self.mswg_test_wrap = MultiSignalWindowGenerator(
            signals=[SIGNAL_CHUNKS, AUX_CHUNKS],
            batch_size=2,
            window_sizes=[4, 2],
            wrap_samples=True)

    def test_len(self):
        self.assertEqual(len(self.mswg_train), 12)
        self.assertEqual(len(self.mswg_train_wrap), 12)
        self.assertEqual(len(self.mswg_test), 12)
        self.assertEqual(len(self.mswg_test_wrap), 12)

    def test_getitem_train_0(self):
        windows, labels = self.mswg_train[0]
        signal_windows = windows[0]
        aux_windows = windows[1]
        self.assertListEqual(labels, [1, 1])
        npt.assert_array_equal(signal_windows[0], [0.9, 0.4, 0.1, 0.2])
        npt.assert_array_equal(signal_windows[1], [0.4, 0.1, 0.2, 0.6])
        npt.assert_array_equal(aux_windows[0], [0.4, 0.2])
        npt.assert_array_equal(aux_windows[1], [0.4, 0.2])

    def test_getitem_train_1(self):
        windows, labels = self.mswg_train[1]
        signal_windows = windows[0]
        aux_windows = windows[1]
        self.assertListEqual(labels, [0, 0])
        npt.assert_array_equal(signal_windows[0], [0.1, 0.2, 0.6, 0.0])
        npt.assert_array_equal(signal_windows[1], [0.2, 0.6, 0.0, 0.3])
        npt.assert_array_equal(aux_windows[0], [0.2, 0.0])
        npt.assert_array_equal(aux_windows[1], [0.2, 0.0])

    def test_getitem_train_wrap_0(self):
        windows, labels = self.mswg_train_wrap[0]
        signal_windows = windows[0]
        aux_windows = windows[1]
        self.assertListEqual(labels, [1, 1])
        npt.assert_array_equal(signal_windows[0], [[0.9], [0.4], [0.1], [0.2]])
        npt.assert_array_equal(signal_windows[1], [[0.4], [0.1], [0.2], [0.6]])
        npt.assert_array_equal(aux_windows[0], [[0.4], [0.2]])
        npt.assert_array_equal(aux_windows[1], [[0.4], [0.2]])

    def test_getitem_train_wrap_1(self):
        windows, labels = self.mswg_train_wrap[1]
        signal_windows = windows[0]
        aux_windows = windows[1]
        self.assertListEqual(labels, [0, 0])
        npt.assert_array_equal(signal_windows[0], [[0.1], [0.2], [0.6], [0.0]])
        npt.assert_array_equal(signal_windows[1], [[0.2], [0.6], [0.0], [0.3]])
        npt.assert_array_equal(aux_windows[0], [[0.2], [0.0]])
        npt.assert_array_equal(aux_windows[1], [[0.2], [0.0]])

    def test_getitem_test_0(self):
        windows = self.mswg_test[0]
        signal_windows = windows[0]
        aux_windows = windows[1]
        npt.assert_array_equal(signal_windows[0], [0.9, 0.4, 0.1, 0.2])
        npt.assert_array_equal(signal_windows[1], [0.4, 0.1, 0.2, 0.6])
        npt.assert_array_equal(aux_windows[0], [0.4, 0.2])
        npt.assert_array_equal(aux_windows[1], [0.4, 0.2])

    def test_getitem_test_1(self):
        windows = self.mswg_test[1]
        signal_windows = windows[0]
        aux_windows = windows[1]
        npt.assert_array_equal(signal_windows[0], [0.1, 0.2, 0.6, 0.0])
        npt.assert_array_equal(signal_windows[1], [0.2, 0.6, 0.0, 0.3])
        npt.assert_array_equal(aux_windows[0], [0.2, 0.0])
        npt.assert_array_equal(aux_windows[1], [0.2, 0.0])

    def test_getitem_test_wrap_0(self):
        windows = self.mswg_test_wrap[0]
        signal_windows = windows[0]
        aux_windows = windows[1]
        npt.assert_array_equal(signal_windows[0], [[0.9], [0.4], [0.1], [0.2]])
        npt.assert_array_equal(signal_windows[1], [[0.4], [0.1], [0.2], [0.6]])
        npt.assert_array_equal(aux_windows[0], [[0.4], [0.2]])
        npt.assert_array_equal(aux_windows[1], [[0.4], [0.2]])

    def test_getitem_test_wrap_1(self):
        windows = self.mswg_test_wrap[1]
        signal_windows = windows[0]
        aux_windows = windows[1]
        npt.assert_array_equal(signal_windows[0], [[0.1], [0.2], [0.6], [0.0]])
        npt.assert_array_equal(signal_windows[1], [[0.2], [0.6], [0.0], [0.3]])
        npt.assert_array_equal(aux_windows[0], [[0.2], [0.0]])
        npt.assert_array_equal(aux_windows[1], [[0.2], [0.0]])
