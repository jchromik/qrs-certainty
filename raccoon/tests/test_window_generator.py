import unittest

from raccoon.generators import WindowGenerator

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]


class TestWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.swg = WindowGenerator(
            SIGNAL_CHUNKS, batch_size=2, window_size=4)

        self.swg_wrap = WindowGenerator(
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
