import numpy as np
import json
import time
import pickle

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [2 * np.random.random((layers[i], layers[i + 1])) - 1 for i in range(len(layers) - 1)]
        self.biases = [np.random.rand(1, layers[i + 1]) for i in range(len(layers) - 1)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward_propagation(self, input_layer):
        hidden_layers = [input_layer]
        for j in range(len(self.weights)):
            hidden_layers.append(self.sigmoid(np.dot(hidden_layers[-1], self.weights[j]) + self.biases[j]))
        return hidden_layers

    def train(self, X_train, y_train, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            for i in range(len(X_train)):
                # Forward propagation
                input_layer = X_train[i].reshape(1, -1)
                hidden_layers = self.forward_propagation(input_layer)
                
                # Calculate the error
                error = y_train[i] - hidden_layers[-1]

                # Backward propagation (simplest form of backpropagation)
                adjustments = [error * self.sigmoid_derivative(hidden_layers[-1])]
                for j in range(len(self.weights) - 1, 0, -1):
                    adjustments.append((adjustments[-1].dot(self.weights[j].T)) * self.sigmoid_derivative(hidden_layers[j]))
                adjustments.reverse()

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[j] += hidden_layers[j].T.dot(adjustments[j])
                    self.biases[j] += np.sum(adjustments[j], axis=0, keepdims=True)

                if time.time() - start_time > 1:
                    print(f'Progress: {(i+epoch*len(X_train)) / (len(X_train)*epochs) * 100}%')
                    start_time = time.time()

    def evaluate(self, X_test, y_test):
        correct_predictions = 0
        for i in range(len(X_test)):
            input_layer = X_test[i].reshape(1, -1)
            hidden_layers = self.forward_propagation(input_layer)
            prediction = np.argmax(hidden_layers[-1])
            if prediction == np.argmax(y_test[i]):
                correct_predictions += 1
        accuracy = correct_predictions / len(X_test) * 100
        print(f'Accuracy on test data: {accuracy}%')

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.layers, self.weights, self.biases), f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            layers, weights, biases = pickle.load(f)
        nn = cls(layers)
        nn.weights = weights
        nn.biases = biases
        return nn

# Load the data from the JSON file
with open('mnist_data_small.json', 'r') as f:
    data = json.load(f)

# Convert lists back to numpy arrays
X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])

num_pixels = X_train.shape[1]
num_classes = y_test.shape[1]

# Define the structure of the neural network
layers = [num_pixels, 128, 64, num_classes]  # You can modify this list to add or remove layers

# Initialize and train the neural network
nn = NeuralNetwork(layers)
# nn = NeuralNetwork.load("trained_model.pkl")
nn.train(X_train, y_train, epochs=10)

# Evaluate the model on the test data
nn.evaluate(X_test, y_test)
nn.save('trained_model2.pkl')

