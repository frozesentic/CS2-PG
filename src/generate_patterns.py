import numpy as np
import os
import cv2
import tensorflow as tf

# Create a directory to store generated patterns
output_directory = 'generated'
os.makedirs(output_directory, exist_ok=True)

# Load the trained generator model
generator = tf.keras.models.load_model('generator_model.h5')

# Generate and save patterns
num_patterns_to_generate = 10
for i in range(num_patterns_to_generate):
    noise = np.random.normal(0, 1, (1, 100))  # Adjust the noise vector size to 100
    generated_pattern = generator.predict(noise)

    # Denormalize pixel values to [0, 255]
    generated_pattern = (generated_pattern * 255).astype(np.uint8)

    # Resize the generated pattern to 400x400
    generated_pattern = cv2.resize(generated_pattern[0], (400, 400))

    # Save the generated pattern
    output_path = os.path.join(output_directory, f'generated_pattern_{i}.png')
    cv2.imwrite(output_path, generated_pattern)
