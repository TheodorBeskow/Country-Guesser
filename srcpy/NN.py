import numpy as np
import json
import time
import pickle

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)


class ConvolutionalNeuralNetwork:
    def __init__(self, clayers, layers):
        self.layers = layers
        self.weights = [2 * np.random.random((layers[i], layers[i + 1])) - 1 for i in range(len(layers) - 1)]
        self.biases = [np.random.rand(1, layers[i + 1]) for i in range(len(layers) - 1)]


class ConvLayer:
    def __init__(self, num_filters, filter_size, height, width, image):
        self.height = height
        self.width = width
        self.image = image
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size * filter_size) # normalize the values
        # self.conv_filter = (2 * np.random.rand(num_filters, filter_size, filter_size) - 1)/(filter_size * filter_size)

    def conv_region(self):
        for j in range(self.height - self.filter_size + 1):
            for k in range(self.width - self.filter_size + 1):
                image_patch = self.image[j:(j + self.filter_size), k:(k + self.filter_size)]
                yield image_patch, j, k

    def forward_prop(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))

        for image_patch, i, j in self.image_region(image):
            conv_out[i, j] = np.sum(image_patch * self.conv_filter, axis=(1, 2))
        return conv_out

    def back_prop(self, dl_dout, learning_rate):
        dl_df_params = np.zeros(self.conv_filter.shape)

        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dl_df_params[k] += image_patch * dl_dout[i, j, k]

        # update our filter values
        self.conv_filter -= learning_rate * dl_df_params
        return dl_df_params


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [2 * np.random.random((layers[i], layers[i + 1])) - 1 for i in range(len(layers) - 1)]
        self.biases = [np.random.rand(1, layers[i + 1]) for i in range(len(layers) - 1)]


    def forward_propagation(self, input_layer):
        hidden_layers = [input_layer]
        for j in range(len(self.weights)):
            hidden_layers.append(ActivationFunction.sigmoid(np.dot(hidden_layers[-1], self.weights[j]) + self.biases[j]))
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
                adjustments = [error * ActivationFunction.sigmoid_derivative(hidden_layers[-1])]
                for j in range(len(self.weights) - 1, 0, -1):
                    adjustments.append((adjustments[-1].dot(self.weights[j].T)) * ActivationFunction.sigmoid_derivative(hidden_layers[j]))
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
with open('Data\mnist_data_small.json', 'r') as f:
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
nn.train(X_train, y_train, epochs=1)

# Evaluate the model on the test data
nn.evaluate(X_test, y_test)
# nn.save('trained_model2.pkl')

"""
import numpy as np
import math

class Convolution:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

class MaxPool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // self.filter_size
        new_w = w // self.filter_size

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.filter_size):(i * self.filter_size + self.filter_size), 
                                  (j * self.filter_size):(j * self.filter_size + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // self.filter_size, w // self.filter_size, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        input = input.flatten()

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        exp = np.exp(totals)
        return exp / np.sum(exp)

# RGB Image with shape (32x32x3)
image = np.random.rand(32*32*3).reshape(32, 32 ,3)

# Convert RGB to grayscale
image_gray = np.mean(image,axis=2)

# Initialize layers
convolution_layer_1 = Convolution(8 ,3)
maxpool_layer_1     = MaxPool(2)
softmax_layer_1     = Softmax(13*13*8 ,10)

# Forward pass through the network
output_convolution_1  = convolution_layer_1.forward(image_gray)
output_maxpool_1      = maxpool_layer_1.forward(output_convolution_1)
output_softmax_1      = softmax_layer_1.forward(output_maxpool_1)

# Print the prediction of the network
print("The network's prediction is: ",np.argmax(output_softmax_1))

"""