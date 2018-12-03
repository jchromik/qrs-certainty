import sys
sys.path.append("..")

import unittest
from collections import namedtuple
from raccoon.utils import annotationutils as au

AnnotationMock = namedtuple('AnnotationMock', ['sample', 'symbol'])

class TestAnnotationUtils(unittest.TestCase):

    def test_extract(self):
        mock_annotation = AnnotationMock([7, 15, 29], ['N', 'L', '|'])
        positions, labels = au._extract(mock_annotation)
        self.assertListEqual(positions, [15, 29])
        self.assertListEqual(labels, ['L', '|'])

    def test_filter(self):
        pos, lab = [7, 15, 29], ['N', '~', '|']

        fpos, flab = au._filter(pos, lab)
        self.assertListEqual(fpos, [7, 29])
        self.assertListEqual(flab, ['N', '|'])

        fpos, flab = au._filter(pos, lab, keep=['N', 'M'])
        self.assertListEqual(fpos, [7])
        self.assertListEqual(flab, ['N'])

    def test_trigger_positons(self):
        mock_annotation = AnnotationMock([7, 15, 29], ['N', '~', '|'])
        self.assertListEqual(au.trigger_points(mock_annotation), [29])