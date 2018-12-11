import unittest

import raccoon.utils.indexutils as iu


class TestIndexUtils(unittest.TestCase):

    def test_rescale(self):
        # TODO: Write test!
        pass

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
        # TODO: Write test!
        pass

    def test_index_pairs_for_batch(self):
        # TODO: Write test!
        pass