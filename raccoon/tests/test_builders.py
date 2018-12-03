import sys
sys.path.append("..")

import unittest
from raccoon.utils import builders

class TestNameBuilder(unittest.TestCase):

    def setUp(self):
        self.name_builder = builders.NameBuilder()

    def test_already_in_use_initially_empty(self):
        self.assertListEqual(self.name_builder.already_in_use, [])

    def test_name_correctly_composed(self):
        name = self.name_builder.name()
        self.assertIn('_', name)
        adjective, animal = tuple(name.split('_'))
        self.assertIn(adjective, builders.ADJECTIVES)
        self.assertIn(animal, builders.ANIMALS)

    def test_name_added_to_already_in_use(self):
        name1 = self.name_builder.name()
        self.assertListEqual(self.name_builder.already_in_use, [name1])
        name2 = self.name_builder.name()
        self.assertListEqual(self.name_builder.already_in_use, [name1, name2])
        self.assertNotEqual(name1, name2)
