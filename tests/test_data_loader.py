import unittest
from scripts.data_loader import load_raw_data, load_all_raw_data

class TestDataLoader(unittest.TestCase):

    def test_load_raw_data(self):
        df = load_raw_data('data/raw/sample_data.csv')
        self.assertEqual(df.shape[0], 100)  # Assuming the file contains 100 rows
        
    def test_load_all_raw_data(self):
        data_frames = load_all_raw_data('data/raw')
        self.assertGreater(len(data_frames), 0)  # Ensure at least one file is loaded

if __name__ == '__main__':
    unittest.main()
