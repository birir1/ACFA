import unittest
from scripts.train_model import train_model, evaluate_model, save_model
from scripts.preprocessing import preprocess_data, split_data

class TestModel(unittest.TestCase):

    def test_train_model(self):
        df = preprocess_data(pd.read_csv('data/processed/Altered-Easy/sample_data.csv'))
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model)  # Ensure the model is trained
        
    def test_model_evaluation(self):
        df = preprocess_data(pd.read_csv('data/processed/Altered-Easy/sample_data.csv'))
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)  # Ensure no errors during evaluation

if __name__ == '__main__':
    unittest.main()
