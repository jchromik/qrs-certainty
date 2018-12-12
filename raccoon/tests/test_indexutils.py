import unittest

import raccoon.utils.indexutils as iu


class TestIndexUtils(unittest.TestCase):

    def test_rescale(self):
        """Chunk index always stays the same. Window index gets adjusted
        according to ratio of window sizes.
        """
        self.assertListEqual(
            iu.rescale(
                index_pairs=[(1, 3), (5, 2), (7, 1)],
                old_window_size=4, new_window_size=2),
            [(1, 1), (5, 1), (7, 0)])

    def test_index_pair(self):
        self.assertTupleEqual(iu.index_pair(0, 4, [10, 12, 11]), (0, 0))
        self.assertTupleEqual(iu.index_pair(1, 4, [10, 12, 11]), (0, 1))
        self.assertTupleEqual(iu.index_pair(6, 4, [10, 12, 11]), (0, 6))
        self.assertTupleEqual(iu.index_pair(7, 4, [10, 12, 11]), (1, 0))
        self.assertTupleEqual(iu.index_pair(15, 4, [10, 12, 11]), (1, 8))
        self.assertTupleEqual(iu.index_pair(16, 4, [10, 12, 11]), (2, 0))
        self.assertTupleEqual(iu.index_pair(23, 4, [10, 12, 11]), (2, 7))
        with self.assertRaises(IndexError):
            iu.index_pair(-1, 4, [10, 12, 11])
        with self.assertRaises(IndexError):
            iu.index_pair(24, 4, [10, 12, 11])

    def test_indexes_for_batch(self):
        self.assertListEqual(
            list(iu.indexes_for_batch(batch_index=1, batch_size=4)),
            [4, 5, 6, 7])
        self.assertListEqual(
            list(iu.indexes_for_batch(batch_index=2, batch_size=4)),
            [8, 9, 10, 11])
        self.assertListEqual(
            list(iu.indexes_for_batch(batch_index=2, batch_size=2)),
            [4, 5])

    def test_index_pairs_for_batch(self):
        self.assertListEqual(
            iu.index_pairs_for_batch(
                batch_index=1,
                batch_size=4,
                window_size=2,
                chunk_sizes=[10, 12, 11]),
            [(0, 4), (0, 5), (0, 6), (0, 7)])
        self.assertListEqual(
            iu.index_pairs_for_batch(
                batch_index=2,
                batch_size=4,
                window_size=2,
                chunk_sizes=[10, 12, 11]),
            [(0, 8), (1, 0), (1, 1), (1, 2)])
