from raccoon.detectors import SingleSignalWindowGenerator

import numpy as np
import unittest

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]

class TestSingleSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.train_gen = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size = 2, window_size = 4,
            trigger_chunks = TRIGGER_CHUNKS, detection_size = 1)

        self.test_gen = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size = 2, window_size = 4)

        self.wrap_train_gen = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size = 2, window_size = 4,
            trigger_chunks = TRIGGER_CHUNKS, detection_size = 1,
            wrap_samples = True)

        self.wrap_test_gen = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size = 2, window_size = 4,
            wrap_samples = True)

    def test_index_pair(self):
        self.assertTupleEqual(self.train_gen._index_pair(0), (0,0))
        self.assertTupleEqual(self.train_gen._index_pair(1), (0,1))
        self.assertTupleEqual(self.train_gen._index_pair(6), (0,6))
        self.assertTupleEqual(self.train_gen._index_pair(7), (1,0))
        self.assertTupleEqual(self.train_gen._index_pair(15), (1,8))
        self.assertTupleEqual(self.train_gen._index_pair(16), (2,0))
        self.assertTupleEqual(self.train_gen._index_pair(23), (2,7))
        with self.assertRaises(IndexError):
            self.train_gen._index_pair(24)
    
    def test_index_pairs_for_batch(self):
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(0),
            [(0, 0), (0, 1)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(1),
            [(0, 2), (0, 3)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(2),
            [(0, 4), (0, 5)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(3),
            [(0, 6), (1, 0)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(7),
            [(1, 7), (1, 8)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(8),
            [(2, 0), (2, 1)])
        self.assertListEqual(
            self.train_gen._index_pairs_for_batch(11),
            [(2, 6), (2, 7)])
        with self.assertRaises(IndexError):
            self.train_gen._index_pairs_for_batch(12)

    def test_window(self):
        self.assertListEqual(self.train_gen._window(0, 0), [0.9, 0.4, 0.1, 0.2])
        self.assertListEqual(self.train_gen._window(0, 1), [0.4, 0.1, 0.2, 0.6])
        self.assertListEqual(self.train_gen._window(0, 6), [0.3, 0.5, 0.0, 0.4])
        self.assertListEqual(self.train_gen._window(1, 0), [0.5, 0.0, 0.2, 0.5])
        self.assertListEqual(self.train_gen._window(1, 8), [0.1, 0.0, 0.7, 0.2])
        self.assertListEqual(self.train_gen._window(2, 0), [0.5, 0.2, 0.9, 0.9])
        self.assertListEqual(self.train_gen._window(2, 7), [0.7, 0.9, 0.1, 0.3])
        with self.assertRaises(IndexError): self.train_gen._window(2, 8)
        with self.assertRaises(IndexError): self.train_gen._window(0, 7)
        with self.assertRaises(IndexError): self.train_gen._window(3, 0)
        with self.assertRaises(IndexError): self.train_gen._window(1, -1)

    def test_windows(self):
        self.assertListEqual(
            self.train_gen._windows([(0, 0), (0, 1), (0, 6), (1, 0)]),
            [ [0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6],
            [0.3, 0.5, 0.0, 0.4], [0.5, 0.0, 0.2, 0.5] ])
        self.assertListEqual(
            self.train_gen._windows([(1, 8), (2, 0), (2, 7)]),
            [ [0.1, 0.0, 0.7, 0.2], [0.5, 0.2, 0.9, 0.9],
            [0.7, 0.9, 0.1, 0.3] ])
        with self.assertRaises(IndexError):
            self.train_gen._windows([(1, 8), (2, 0), (2, 8)])

    def test_label(self):
        self.assertEqual(self.train_gen._label(0, 0), 1)
        self.assertEqual(self.train_gen._label(0, 1), 1)
        self.assertEqual(self.train_gen._label(0, 2), 0)
        self.assertEqual(self.train_gen._label(0, 3), 0)
        self.assertEqual(self.train_gen._label(0, 4), 0)
        self.assertEqual(self.train_gen._label(0, 5), 1)
        self.assertEqual(self.train_gen._label(0, 6), 0)
        self.assertEqual(self.train_gen._label(1, 0), 1)
        self.assertEqual(self.train_gen._label(1, 1), 0)
        with self.assertRaises(IndexError): self.train_gen._label(2, 8)
        with self.assertRaises(IndexError): self.train_gen._label(0, 7)
        with self.assertRaises(IndexError): self.train_gen._label(3, 0)
        with self.assertRaises(IndexError): self.train_gen._label(1, -1)
        with self.assertRaises(RuntimeError): self.test_gen._label(1, 1)

    def test_labels(self):
        self.assertListEqual(
            self.train_gen._labels([(0, 0), (0, 1), (0, 6), (1, 0)]),
            [1, 1, 0, 1])
        self.assertListEqual(
            self.train_gen._labels([(1, 8), (2, 0), (2, 7)]),
            [0, 0, 1])
        with self.assertRaises(IndexError):
            self.train_gen._labels([(1, 8), (2, 0), (2, 8)])
        with self.assertRaises(RuntimeError):
            self.test_gen._labels([(1, 8), (2, 0), (2, 7)])

    def test_window_batch(self):
        self.assertListEqual(
            self.train_gen.window_batch(0),
            [ [0.9, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.6] ])
        self.assertListEqual(
            self.train_gen.window_batch(1),
            [ [0.1, 0.2, 0.6, 0.0], [0.2, 0.6, 0.0, 0.3] ])
        self.assertListEqual(
            self.train_gen.window_batch(3),
            [ [0.3, 0.5, 0.0, 0.4], [0.5, 0.0, 0.2, 0.5] ])
        self.assertListEqual(
            self.train_gen.window_batch(11),
            [ [0.5, 0.7, 0.9, 0.1], [0.7, 0.9, 0.1, 0.3] ])
        with self.assertRaises(IndexError):
            self.train_gen.window_batch(12)

    def test_label_batch(self):
        self.assertListEqual(self.train_gen.label_batch(0), [1, 1])
        self.assertListEqual(self.train_gen.label_batch(1), [0, 0])
        self.assertListEqual(self.train_gen.label_batch(3), [0, 1])
        self.assertListEqual(self.train_gen.label_batch(11), [0, 1])
        with self.assertRaises(IndexError):
            self.train_gen.label_batch(12)
        with self.assertRaises(RuntimeError):
            self.test_gen.label_batch(11)

    def test_train_batch(self):
        for i in range(0, 12):
            self.assertTupleEqual(
                self.train_gen.train_batch(i),
                (self.train_gen.window_batch(i), self.train_gen.label_batch(i)))
        with self.assertRaises(IndexError):
            self.train_gen.train_batch(12)
        with self.assertRaises(RuntimeError):
            self.test_gen.train_batch(11)

    def test_getitem_train(self):
        for i in range(0, 12):
            windows, labels = self.train_gen.train_batch(i)
            np_windows, np_labels = self.train_gen[i]
            np.testing.assert_array_equal(np.array(windows), np_windows)
            np.testing.assert_array_equal(np.array(labels), np_labels)
        with self.assertRaises(IndexError):
            self.train_gen[12]

    def test_getitem_wrap_train(self):
        for i in range(0, 12):
            windows, labels = self.wrap_train_gen.train_batch(i)
            np_windows, np_labels = self.wrap_train_gen[i]
            self.assertTupleEqual(np_windows.shape, (2, 4, 1))
            np.testing.assert_array_equal(
                np.array(windows).reshape(2, 4, 1),
                np_windows)
            np.testing.assert_array_equal(np.array(labels), np_labels)
        with self.assertRaises(IndexError):
            self.train_gen[12]

    def test_getitem_test(self):
        for i in range(0, 12):
            windows = self.test_gen.window_batch(i)
            np_windows = self.test_gen[i]
            np.testing.assert_array_equal(np.array(windows), np_windows)
        with self.assertRaises(IndexError):
            self.test_gen[12]

    def test_getitem_wrap_test(self):
        for i in range(0, 12):
            windows = self.wrap_test_gen.window_batch(i)
            np_windows = self.wrap_test_gen[i]
            self.assertTupleEqual(np_windows.shape, (2, 4, 1))
            np.testing.assert_array_equal(
                np.array(windows).reshape(2, 4, 1),
                np_windows)
        with self.assertRaises(IndexError):
            self.test_gen[12]

    def test_len(self):
        self.assertEqual(len(self.train_gen), 12)

