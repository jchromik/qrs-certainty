import unittest

import numpy as np
import numpy.testing as npt

from raccoon.generators import (
    SingleSignalWindowGenerator, WindowGenerator, LabelGenerator)

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]


class TestSingleSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.windows = WindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4)

        self.windows_wrap = WindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4, wrap_samples=True)

        self.labels = LabelGenerator(
            TRIGGER_CHUNKS, [len(chunk) for chunk in SIGNAL_CHUNKS],
            batch_size=2, window_size=4, detection_size=1)

        self.sswg_train = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4,
            trigger_chunks=TRIGGER_CHUNKS, detection_size=1)

        self.sswg_test = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4)

        self.sswg_train_wrap = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4,
            trigger_chunks=TRIGGER_CHUNKS, detection_size=1,
            wrap_samples=True)

        self.sswg_test_wrap = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4,
            wrap_samples=True)

    def test_getitem_train(self):
        for i in range(0, 12):
            windows = self.windows.batch(i)
            labels = self.labels[i]
            np_windows, np_labels = self.sswg_train[i]
            npt.assert_array_equal(np.array(windows), np_windows)
            npt.assert_array_equal(np.array(labels), np_labels)
        with self.assertRaises(IndexError):
            self.sswg_train[12]

    def test_getitem_wrap_train(self):
        for i in range(0, 12):
            windows = self.windows_wrap.batch(i)
            labels = self.labels[i]
            np_windows, np_labels = self.sswg_train_wrap[i]
            self.assertTupleEqual(np_windows.shape, (2, 4, 1))
            npt.assert_array_equal(
                np.array(windows).reshape(2, 4, 1),
                np_windows)
            npt.assert_array_equal(np.array(labels), np_labels)
        with self.assertRaises(IndexError):
            self.sswg_train[12]

    def test_getitem_test(self):
        for i in range(0, 12):
            windows = self.windows.batch(i)
            np_windows = self.sswg_test[i]
            npt.assert_array_equal(np.array(windows), np_windows)
        with self.assertRaises(IndexError):
            self.sswg_test[12]

    def test_getitem_wrap_test(self):
        for i in range(0, 12):
            windows = self.windows_wrap.batch(i)
            np_windows = self.sswg_test_wrap[i]
            self.assertTupleEqual(np_windows.shape, (2, 4, 1))
            npt.assert_array_equal(
                np.array(windows).reshape(2, 4, 1),
                np_windows)
        with self.assertRaises(IndexError):
            self.sswg_test[12]

    def test_len(self):
        self.assertEqual(len(self.sswg_train), 12)
