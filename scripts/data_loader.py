import os
import pandas as pd

def load_raw_data(file_path):
    """
    Loads raw data from a given file path (e.g., CSV, JSON).
    :param file_path: Path to the raw data file.
    :return: DataFrame containing the raw data.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format: Only CSV and JSON are supported.")

def load_all_raw_data(data_directory):
    """
    Loads all raw data files from the specified directory.
    :param data_directory: Directory containing raw data files.
    :return: List of DataFrames.
    """
    raw_data_files = [f for f in os.listdir(data_directory) if f.endswith('.csv') or f.endswith('.json')]
    data_frames = []
    for file in raw_data_files:
        file_path = os.path.join(data_directory, file)
        data_frames.append(load_raw_data(file_path))
    return data_frames
