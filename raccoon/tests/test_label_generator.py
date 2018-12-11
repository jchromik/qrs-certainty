import unittest

from raccoon.generators import LabelGenerator

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]


class TestLabelGenerator(unittest.TestCase):

    def setUp(self):
        self.labels = LabelGenerator(
            TRIGGER_CHUNKS, [len(chunk) for chunk in SIGNAL_CHUNKS],
            batch_size=2, window_size=4, detection_size=1)

    def test_index_pair(self):
        self.assertTupleEqual(self.labels.index_pair(0), (0, 0))
        self.assertTupleEqual(self.labels.index_pair(1), (0, 1))
        self.assertTupleEqual(self.labels.index_pair(6), (0, 6))
        self.assertTupleEqual(self.labels.index_pair(7), (1, 0))
        self.assertTupleEqual(self.labels.index_pair(23), (2, 7))
        with self.assertRaises(IndexError):
            self.labels.index_pair(-1)
        with self.assertRaises(IndexError):
            self.labels.index_pair(24)

    def test_index_pairs_for_batch(self):
        self.assertListEqual(
            self.labels.index_pairs_for_batch(0),
            [(0, 0), (0, 1)])
        self.assertListEqual(
            self.labels.index_pairs_for_batch(3),
            [(0, 6), (1, 0)])
        self.assertListEqual(
            self.labels.index_pairs_for_batch(11),
            [(2, 6), (2, 7)])
        with self.assertRaises(IndexError):
            self.labels.index_pairs_for_batch(12)

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

    def test_len(self):
        self.assertEqual(len(self.labels), 12)