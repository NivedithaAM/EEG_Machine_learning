import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
import pywt
from glob import glob

# Set path for EEG datasets
DATA_DIR = "C:/Users/Dell/Desktop/eeg_datasets"

# Find all dataset files
file_paths = sorted(glob(os.path.join(DATA_DIR, "*.csv")))

# Separate noisy and clean files
clean_files = [f for f in file_paths if "_noisy" not in f]
noisy_files = [f.replace(".csv", "_noisy.csv") for f in clean_files if f.replace(".csv", "_noisy.csv") in file_paths]

# Data Normalization Function
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-8), mean, std

# Modified Batch Generator with fixed length
def batch_generator(noisy_files, clean_files, batch_size=5, fixed_length=31000):
    while True:
        for i in range(0, len(noisy_files), batch_size):
            batch_noisy = []
            batch_clean = []
            for j in range(batch_size):
                if i + j >= len(noisy_files):
                    break
                
                # Load data
                noisy_data = pd.read_csv(noisy_files[i + j]).values
                clean_data = pd.read_csv(clean_files[i + j]).values
                
                # Pad or truncate to fixed length
                def adjust_length(data, length):
                    if data.shape[0] > length:
                        return data[:length]
                    elif data.shape[0] < length:
                        return np.pad(data, ((0, length - data.shape[0]), (0, 0)), mode='constant')
                    return data
                
                noisy_data = adjust_length(noisy_data, fixed_length)
                clean_data = adjust_length(clean_data, fixed_length)
                
                # Normalize
                noisy_data, _, _ = normalize_data(noisy_data)
                clean_data, _, _ = normalize_data(clean_data)
                
                batch_noisy.append(noisy_data)
                batch_clean.append(clean_data)
            
            yield np.array(batch_noisy), np.array(batch_clean)

# Fixed Model Architecture with matching dimensions
def build_autoencoder(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)  # 31000 -> 15500
    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)  # 15500 -> 7750
    
    # Decoder
    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(size=2)(x)  # 7750 -> 15500
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(size=2)(x)  # 15500 -> 31000
    outputs = tf.keras.layers.Conv1D(input_shape[-1], kernel_size=5, activation='linear', padding='same')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Fixed parameters
fixed_length = 31000
input_shape = (fixed_length, 19)  # Assuming 19 channels

# Build model
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

# Train model
history = autoencoder.fit(
    batch_generator(noisy_files, clean_files, batch_size=5, fixed_length=fixed_length),
    steps_per_epoch=len(noisy_files)//5,
    epochs=25
)

# Select a custom noisy signal file
custom_noisy_file = "C:/Users/Dell/Desktop/eeg_datasets/s10_noisy.csv"  
custom_clean_file = "C:/Users/Dell/Desktop/eeg_datasets/s10.csv"  # Corresponding clean file

# Load the noisy and clean signals
custom_noisy = pd.read_csv(custom_noisy_file).values
custom_clean = pd.read_csv(custom_clean_file).values

# Ensure both signals have the same length
fixed_length = 31000
def adjust_length(data, length):
    if data.shape[0] > length:
        return data[:length]
    elif data.shape[0] < length:
        return np.pad(data, ((0, length - data.shape[0]), (0, 0)), mode='constant')
    return data

custom_noisy = adjust_length(custom_noisy, fixed_length)
custom_clean = adjust_length(custom_clean, fixed_length)

# Normalize using the same mean and std
custom_noisy_norm, mean, std = normalize_data(custom_noisy)
custom_clean_norm, _, _ = normalize_data(custom_clean)

# Predict using trained model
denoised_signal_norm = autoencoder.predict(np.expand_dims(custom_noisy_norm, axis=0))[0]

# Denormalize the output signal
denoised_signal = denoised_signal_norm * std + mean

# Plot Noisy vs Denoised vs Clean Signal
plt.figure(figsize=(12, 6))
plt.plot(custom_noisy[:, 0], color='r', alpha=0.5, label="Noisy Signal")
plt.plot(denoised_signal[:, 0], color='b', alpha=0.8, label="Denoised Signal")
plt.plot(custom_clean[:, 0], color='g', alpha=0.7, label="Clean Signal (Ground Truth)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Custom EEG Signal Denoising - Channel 1")
plt.show()
