import unittest
from dynamic_fmap.core import DynamicFeatureMapper

class TestDynamicFeatureMapper(unittest.TestCase):
    def test_map(self):
        mapper = DynamicFeatureMapper()
        self.assertEqual(mapper.map([1, 2, 3]), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()