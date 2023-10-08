import os
import cv2
import numpy as np

# Define the path to your dataset directory
dataset_path = 'dataset'

# Define parameters for data preprocessing
image_size = (128, 128)
output_directory = 'preprocessed_dataset'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Preprocess the images and save them to the output directory
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    output_category_path = os.path.join(output_directory, category)
    os.makedirs(output_category_path, exist_ok=True)

    for image_file in os.listdir(category_path):
        image_path = os.path.join(category_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
        image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

        output_image_path = os.path.join(output_category_path, image_file)
        cv2.imwrite(output_image_path, image * 255.0)
