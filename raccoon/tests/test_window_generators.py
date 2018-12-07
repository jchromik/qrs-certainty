import unittest

from raccoon.detectors import SingleSignalWindowGenerator

SIGNAL_CHUNKS = [
    [0.9, 0.4, 0.1, 0.2, 0.6, 0.0, 0.3, 0.5, 0.0, 0.4],
    [0.5, 0.0, 0.2, 0.5, 0.8, 0.4, 0.2, 0.4, 0.1, 0.0, 0.7, 0.2],
    [0.5, 0.2, 0.9, 0.9, 0.0, 0.9, 0.5, 0.7, 0.9, 0.1, 0.3]]

TRIGGER_CHUNKS = [[2, 3, 7], [2, 5, 7], [1, 7, 9]]

class TestSingleSignalWindowGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = SingleSignalWindowGenerator(
            SIGNAL_CHUNKS, batch_size = 2, window_size = 4,
            trigger_chunks = TRIGGER_CHUNKS, detection_size = 3)

    def test_index_pair(self):
        self.assertTupleEqual(self.gen._index_pair(0), (0,0))
    
    def test_index_pairs_for_batch(self):
        pass

    def test_window(self):
        pass

    def test_windows(self):
        pass

    def test_label(self):
        pass

    def test_labels(self):
        pass

    def test_window_batch(self):
        pass

    def test_label_batch(self):
        pass

    def test_batch(self):
        pass

    def test_getitem(self):
        pass

    def test_len(self):
        pass