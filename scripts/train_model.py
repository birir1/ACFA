import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scripts.model import create_model
import os

# Load your dataset here
def load_data():
    # Replace this with actual loading from disk or dataset folder
    # Example dummy grayscale data
    x_train = np.random.rand(300, 224, 224, 1).astype('float32')  # 300 grayscale images
    y_train = np.random.randint(0, 3, size=(300,))  # 3 classes
    y_train = to_categorical(y_train, num_classes=3)
    return x_train, y_train

# Load training data
x_train, y_train = load_data()

# Data augmentation (optional but good for training generalization)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Create and compile model
model = create_model(input_shape=(224, 224, 1))
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=10, steps_per_epoch=len(x_train)//16)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/auth_model.h5")
print("✅ Model training complete and saved to 'models/auth_model.h5'")
