import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the raw data by cleaning and handling missing values.
    :param df: DataFrame containing raw data.
    :return: Cleaned DataFrame.
    """
    df = df.dropna()  # Drop rows with missing values
    return df

def feature_engineering(df):
    """
    Example feature engineering: Create a new column based on existing data.
    :param df: DataFrame containing data.
    :return: DataFrame with new features.
    """
    df['new_feature'] = df['feature_1'] * df['feature_2']  # Placeholder for actual feature engineering
    return df

def split_data(df, test_size=0.2):
    """
    Split data into training and testing sets.
    :param df: DataFrame containing the data.
    :param test_size: The size of the test data.
    :return: X_train, X_test, y_train, y_test.
    """
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=42)

def scale_data(X_train, X_test):
    """
    Scales the features to a standard range.
    :param X_train: Training features.
    :param X_test: Testing features.
    :return: Scaled features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
