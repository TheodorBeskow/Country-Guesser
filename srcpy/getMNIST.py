import numpy as np
import json
from keras.datasets import mnist
from keras.utils import np_utils

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28*28 images to a 784 vector for each image and normalize
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32') / 255

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Only save a tenth of the data
X_train = X_train[:len(X_train)//10]
y_train = y_train[:len(y_train)//10]
X_test = X_test[:len(X_test)//10]
y_test = y_test[:len(y_test)//10]

# Convert numpy arrays to lists for JSON serialization
X_train = X_train.tolist()
y_train = y_train.tolist()
X_test = X_test.tolist()
y_test = y_test.tolist()

# Save the data in JSON format
with open('mnist_data_small.json', 'w') as f:
    json.dump({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}, f)
