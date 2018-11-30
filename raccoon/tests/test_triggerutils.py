import sys
sys.path.append("..")

import unittest

from raccoon.utils import triggerutils as tu

class TestTriggerUtils(unittest.TestCase):

    def test_signal_to_spikes(self):
        signal = [0]*10 + [1]*5 + [0]*2 + [1]*7
        self.assertEqual(list(tu.signal_to_spikes(signal)), [(10,15), (17,24)])

    def test_spikes_to_points(self):
        spikes = [(25,50), (2,7), (10,20)]
        points = [37, 4, 15]
        self.assertListEqual(tu.spikes_to_points(spikes), points)

    def test_points_to_signal(self):
        points = [2, 7, 13]
        signal_length = 20
        signals = [
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
            [0.,1.,1.,0.,0.,0.,1.,1.,0.,0.,0.,0.,1.,1.,0.,0.,0.,0.,0.,0.],
            [0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.],
            [1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,0.,1.,1.,1.,1.,0.,0.,0.,0.,0.],
            [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,1.,0.,0.,0.,0.],
            [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.]
        ]
        for window_size in range(0,len(signals)):
            self.assertListEqual(
                tu.points_to_signal(points, signal_length, window_size),
                signals[window_size])

    def test_spikes_to_certainties(self):
        signal = [0.0, 0.5, 1.0, 1.0, 0.2, 0.0, 0.7, 0.5]
        spikes = [(0,2), (1,5), (6,8)]
        certainties = [0.25, 0.675, 0.6]
        self.assertListEqual(
            tu.spikes_to_certainties(spikes, signal),
            certainties)