import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Sample dataset
data = pd.read_csv('all_data.csv')

df = pd.DataFrame(data)

# Normalize the data
data = df.values
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Define the generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=10),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])
    return model

# Define the discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=10),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile the GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

gan_input = layers.Input(shape=(10,))
generated_data = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, 10))
        fake_data = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 10))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss_real + d_loss_fake}, Generator Loss: {g_loss}")

train_gan(gan, generator, discriminator, data)

# Generate synthetic data
noise = np.random.normal(0, 1, (3000000, 10))
synthetic_data = generator.predict(noise)

# Convert synthetic data back to original scale and data types
synthetic_data = synthetic_data * (np.max(df.values, axis=0) - np.min(df.values, axis=0)) + np.min(df.values, axis=0)
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
synthetic_df = synthetic_df.astype(df.dtypes)

print("Synthetic Data:\n", synthetic_df.head())