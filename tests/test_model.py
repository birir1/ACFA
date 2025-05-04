# tests/test_model.py

import pytest
from tensorflow.keras.models import load_model
from scripts.data_loader import load_image_data_from_processed

def test_model_accuracy():
    # Load the trained model
    model = load_model('models/trained_models/fingerprint_model.h5')
    
    # Load test data
    test_images, test_labels = load_image_data_from_processed('C:/Users/BirirM/ACFA/data/processed/Altered-Medium')
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_images, test_labels)
    
    assert accuracy > 0.85, f"Test accuracy is too low: {accuracy}"  # Expecting at least 85% accuracy
