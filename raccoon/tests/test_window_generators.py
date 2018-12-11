import unittest

import numpy as np
import numpy.testing as npt

from raccoon.detectors import (
    SingleSignalWindowGenerator, MultiSignalWindowGenerator,
    SignalWindowGenerator, LabelGenerator)

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

AUX_CHUNKS = [
    [0.4, 0.2, 0.0, 0.5, 0.4],
    [0.0, 0.5, 0.4, 0.4, 0.0, 0.2],
    [0.2, 0.9, 0.9, 0.7, 0.1]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]


class TestSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.swg = SignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4)

        self.swg_wrap = SignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4, wrap_samples=True)

    def test_index_pair(self):
        self.assertTupleEqual(self.swg.index_pair(0), (0, 0))
        self.assertTupleEqual(self.swg.index_pair(1), (0, 1))
        self.assertTupleEqual(self.swg.index_pair(6), (0, 6))
        self.assertTupleEqual(self.swg.index_pair(7), (1, 0))
        self.assertTupleEqual(self.swg.index_pair(15), (1, 8))
        self.assertTupleEqual(self.swg.index_pair(16), (2, 0))
        self.assertTupleEqual(self.swg.index_pair(23), (2, 7))
        with self.assertRaises(IndexError):
            self.swg.index_pair(24)

    def test_index_pairs_for_batch(self):
        self.assertListEqual(
            self.swg.index_pairs_for_batch(0),
            [(0, 0), (0, 1)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(1),
            [(0, 2), (0, 3)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(2),
            [(0, 4), (0, 5)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(3),
            [(0, 6), (1, 0)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(7),
            [(1, 7), (1, 8)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(8),
            [(2, 0), (2, 1)])
        self.assertListEqual(
            self.swg.index_pairs_for_batch(11),
            [(2, 6), (2, 7)])
        with self.assertRaises(IndexError):
            self.swg.index_pairs_for_batch(12)

    def test_window(self):
        self.assertListEqual(self.swg.window(0, 0), [0.9, 0.4, 0.1, 0.2])
        self.assertListEqual(self.swg.window(0, 1), [0.4, 0.1, 0.2, 0.6])
        self.assertListEqual(self.swg.window(0, 6), [0.3, 0.5, 0.0, 0.4])
        self.assertListEqual(self.swg.window(1, 0), [0.5, 0.0, 0.2, 0.5])
        self.assertListEqual(self.swg.window(1, 8), [0.1, 0.0, 0.7, 0.2])
        self.assertListEqual(self.swg.window(2, 0), [0.5, 0.2, 0.9, 0.9])
        self.assertListEqual(self.swg.window(2, 7), [0.7, 0.9, 0.1, 0.3])
        with self.assertRaises(IndexError):
            self.swg.window(2, 8)
        with self.assertRaises(IndexError):
            self.swg.window(0, 7)
        with self.assertRaises(IndexError):
            self.swg.window(3, 0)
        with self.assertRaises(IndexError):
            self.swg.window(1, -1)

    def test_windows(self):
        self.assertListEqual(
            self.swg.windows([(0, 0), (0, 1), (0, 6), (1, 0)]),
            [[0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6],
             [0.3, 0.5, 0.0, 0.4], [0.5, 0.0, 0.2, 0.5]])
        self.assertListEqual(
            self.swg.windows([(1, 8), (2, 0), (2, 7)]),
            [[0.1, 0.0, 0.7, 0.2], [0.5, 0.2, 0.9, 0.9],
             [0.7, 0.9, 0.1, 0.3]])
        with self.assertRaises(IndexError):
            self.swg.windows([(1, 8), (2, 0), (2, 8)])

    def test_batch(self):
        self.assertListEqual(
            self.swg.batch(0),
            [[0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6]])
        self.assertListEqual(
            self.swg.batch(1),
            [[0.1, 0.2, 0.6, 0.0], [0.2, 0.6, 0.0, 0.3]])
        self.assertListEqual(
            self.swg.batch(3),
            [[0.3, 0.5, 0.0, 0.4], [0.5, 0.0, 0.2, 0.5]])
        self.assertListEqual(
            self.swg.batch(11),
            [[0.5, 0.7, 0.9, 0.1], [0.7, 0.9, 0.1, 0.3]])
        with self.assertRaises(IndexError):
            self.swg.batch(12)


class TestLabelGenerator(unittest.TestCase):

    def setUp(self):
        self.labels = LabelGenerator(
            TRIGGER_CHUNKS, [len(chunk) for chunk in SIGNAL_CHUNKS],
            batch_size=2, window_size=4, detection_size=1)

    def test_label(self):
        self.assertEqual(self.labels.label(0, 0), 1)
        self.assertEqual(self.labels.label(0, 1), 1)
        self.assertEqual(self.labels.label(0, 2), 0)
        self.assertEqual(self.labels.label(0, 3), 0)
        self.assertEqual(self.labels.label(0, 4), 0)
        self.assertEqual(self.labels.label(0, 5), 1)
        self.assertEqual(self.labels.label(0, 6), 0)
        self.assertEqual(self.labels.label(1, 0), 1)
        self.assertEqual(self.labels.label(1, 1), 0)
        with self.assertRaises(IndexError):
            self.labels.label(2, 8)
        with self.assertRaises(IndexError):
            self.labels.label(0, 7)
        with self.assertRaises(IndexError):
            self.labels.label(1, -1)
        with self.assertRaises(IndexError):
            self.labels.label(3, 0)
        with self.assertRaises(IndexError):
            self.labels.label(-1, 0)

    def test_labels(self):
        self.assertListEqual(
            self.labels.labels([(0, 0), (0, 1), (0, 6), (1, 0)]),
            [1, 1, 0, 1])
        self.assertListEqual(
            self.labels.labels([(1, 8), (2, 0), (2, 7)]),
            [0, 0, 1])
        with self.assertRaises(IndexError):
            self.labels.labels([(1, 8), (2, 0), (3, 0)])

    def test_getitem(self):
        self.assertListEqual(self.labels[0], [1, 1])
        self.assertListEqual(self.labels[1], [0, 0])
        self.assertListEqual(self.labels[3], [0, 1])
        self.assertListEqual(self.labels[11], [0, 1])
        with self.assertRaises(IndexError):
            self.labels[12]


class TestSingleSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.windows = SignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4)

        self.windows_wrap = SignalWindowGenerator(
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
