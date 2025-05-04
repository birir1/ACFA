from PIL import Image
import os
import numpy as np

def load_image_data_from_processed(processed_path):
    images = []
    labels = []

    # Assuming images are organized in folders corresponding to class labels
    class_folders = os.listdir(processed_path)
    
    for folder in class_folders:
        class_path = os.path.join(processed_path, folder)
        
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.BMP'):  # Only load BMP files
                    img_path = os.path.join(class_path, filename)
                    img = Image.open(img_path).convert('RGB')  # Convert grayscale to RGB
                    img = img.resize((224, 224))  # Resize to match input size (224x224)
                    images.append(np.array(img))
                    labels.append(folder)  # Assuming folder name as the label

    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels
