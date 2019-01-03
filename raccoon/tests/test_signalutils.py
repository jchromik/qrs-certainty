import unittest

import numpy as np
import numpy.testing as npt

import raccoon.utils.signalutils as su


class TestSignalUtils(unittest.TestCase):

    def test_window_average(self):
        """The function window_average should remove samples from the end of the
        signal until its length is a multiple of window_size. Then create
        disjoint windows of size window_size from the signal and compute their
        value average (=mean value).
        """
        npt.assert_array_equal(
            su.window_average(np.array([1, 2, 3, 4, 5]), window_size=2),
            np.array([1.5, 3.5]))
        npt.assert_array_equal(
            su.window_average(np.array([1, 2, 3, 4, 5]), window_size=1),
            np.array([1, 2, 3, 4, 5]))