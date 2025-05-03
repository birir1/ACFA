from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the training data.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data and return a performance report.
    :param model: Trained model.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return: None.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, model_path):
    """
    Save the trained model to a file.
    :param model: Trained model.
    :param model_path: Path to save the model.
    :return: None.
    """
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")
