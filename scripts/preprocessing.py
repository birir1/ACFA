# scripts/preprocessing.py

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(images, labels):
    """
    Perform data preprocessing, such as encoding labels and augmenting data.
    """
    # Encode the labels (assuming categorical classification)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Data augmentation for images
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(images)
    
    return images, labels, datagen
