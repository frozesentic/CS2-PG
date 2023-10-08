import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy

# Force TensorFlow to use CPU only (even if a GPU is available)
tf.config.experimental.set_visible_devices([], 'GPU')

def build_generator():
    model = tf.keras.Sequential([
        Dense(1024, input_dim=100, activation='relu'),
        Dense(400 * 400 * 4, activation='sigmoid'),  # Output shape matches (400, 400, 4)
        Reshape((400, 400, 4)),
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        Flatten(input_shape=(400, 400, 4)),  # Input shape matches generator output
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator during GAN training
    gan_input = tf.keras.layers.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.models.Model(gan_input, gan_output)
    return gan


# Define your real data generator
def load_real_samples(batch_size):
    real_images_dir = 'dataset'

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    real_data_generator = datagen.flow_from_directory(
        real_images_dir,
        target_size=(400, 400),
        batch_size=batch_size,
        class_mode='binary',
    )
    return real_data_generator

# Training loop
epochs = 100  # Set the number of epochs you want
batch_size = 8  # Set to the number of images you have in your dataset

real_samples_generator = load_real_samples(batch_size)

# Define the generator model
generator = build_generator()

# Define the discriminator model
discriminator = build_discriminator()

# Define the GAN model
gan = build_gan(generator, discriminator)

# Compile discriminator and GAN
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Training step with your real data
    real_images, _ = next(real_samples_generator)
    discriminator_loss_real = discriminator.train_on_batch(real_images, tf.ones((batch_size, 1)))

    # Training step with your generated data
    noise = tf.random.normal((batch_size, 100))
    generated_images = generator.predict(noise)
    discriminator_loss_generated = discriminator.train_on_batch(generated_images, tf.zeros((batch_size, 1)))

    # Training step for the GAN
    noise = tf.random.normal((batch_size, 100))
    gan_loss = gan.train_on_batch(noise, tf.ones((batch_size, 1)))

    # Print losses or other training metrics as needed
    print(f"Discriminator Loss Real: {discriminator_loss_real}")
    print(f"Discriminator Loss Generated: {discriminator_loss_generated}")
    print(f"GAN Loss: {gan_loss}")

# Save the trained generator model to a file
generator.save('generator_model.h5')

# Example usage:
batch_size = 16
real_samples_generator = load_real_samples(batch_size)

def train_gan(generator, discriminator, gan, epochs, batch_size):
    # Training loop
    for epoch in range(epochs):
        # Train discriminator
        noise = tf.random.normal((batch_size, 100))
        generated_images = generator.predict(noise)
        real_images = load_real_samples(batch_size)  # Implement this function to load real samples
        discriminator_loss_real = discriminator.train_on_batch(real_images, tf.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(generated_images, tf.zeros((batch_size, 1)))
        discriminator_loss = 0.5 * (discriminator_loss_real[0] + discriminator_loss_fake[0])

        # Train generator (via GAN)
        noise = tf.random.normal((batch_size, 100))
        generator_loss = gan.train_on_batch(noise, tf.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {discriminator_loss}, G Loss: {generator_loss}")

        # Save generator weights at checkpoints (optional)
        if (epoch + 1) % checkpoint_interval == 0:
            generator.save(f'generator_model_epoch_{epoch + 1}.h5')

# Set your training parameters
epochs = 10000
batch_size = 16
checkpoint_interval = 100

# Define the generator model
generator = build_generator()

# Define the discriminator model
discriminator = build_discriminator()

# Define the GAN model
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, epochs, batch_size)
