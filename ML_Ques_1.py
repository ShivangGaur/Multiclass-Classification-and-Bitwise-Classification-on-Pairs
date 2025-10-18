import numpy as np

class NeuralNetwork:
    def _init_(self, layer_dims, learning_rate=0.1):
        """
        Initializes the neural network.
        Arguments:
        layer_dims -- list containing the dimensions of each layer in the network
        learning_rate -- learning rate for gradient descent
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initializes weights and biases for each layer.
        Returns:
        parameters -- dictionary containing initialized weights and biases
        """
        np.random.seed(1)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """Derivative of ReLU activation function."""
        return np.where(Z > 0, 1, 0)

    def softmax(self, Z):
        """Softmax activation function for the output layer."""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """
        Implements forward propagation.
        Arguments:
        X -- input data
        Returns:
        AL -- the output of the last layer (softmax output)
        caches -- list of caches containing information for backward propagation
        """
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            caches.append((A_prev, W, b, Z))

        # Output layer (softmax activation)
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = np.dot(W, A) + b
        AL = self.softmax(Z)
        caches.append((A, W, b, Z))

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Computes the cross-entropy cost.
        Arguments:
        AL -- probability vector from forward propagation (output of softmax)
        Y -- true label vector (one-hot encoded)
        Returns:
        cost -- cross-entropy cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-9)) / m
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, caches):
        """
        Implements backward propagation.
        Arguments:
        AL -- probability vector, output of the forward propagation
        Y -- true label vector (one-hot encoded)
        caches -- list of caches containing information from forward propagation
        Returns:
        grads -- dictionary containing gradients for each layer
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Derivative of cost with respect to AL
        dZ = AL - Y
        A_prev, W, b, Z = caches[L - 1]
        grads[f'dW{L}'] = (1/m) * np.dot(dZ, A_prev.T)
        grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        # Loop over layers in reverse order
        for l in reversed(range(L - 1)):
            A_prev, W, b, Z = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads):
        """
        Updates parameters using gradient descent.
        Arguments:
        grads -- dictionary containing gradients for each layer
        """
        L = len(self.parameters) // 2

        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def train(self, X, Y, num_iterations=1000):
        """
        Trains the neural network.
        Arguments:
        X -- input data
        Y -- true labels (one-hot encoded)
        num_iterations -- number of iterations for training
        Returns:
        costs -- list of costs over the training process
        """
        costs = []

        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads)

            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")

        return costs

    def predict(self, X):
        """
        Predicts labels for input data X.
        Arguments:
        X -- input data
        Returns:
        predictions -- predicted labels for X
        """
        AL, _ = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

    def compute_accuracy(self, predictions, Y):
        """
        Computes the accuracy of predictions.
        Arguments:
        predictions -- predicted labels
        Y -- true labels
        Returns:
        accuracy -- percentage of correct predictions
        """
        Y_true = np.argmax(Y, axis=0)
        return np.mean(predictions == Y_true) * 100
    
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Function to preprocess the MNIST dataset
def preprocess_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    # Flatten the images to turn each 28x28 image into a 784-dimensional vector
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0  # Normalizing by dividing by 255
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    
    # One-hot encode the labels
    enc = OneHotEncoder(sparse_output=False)
    Y_train = enc.fit_transform(Y_train.reshape(-1, 1)).T
    Y_test = enc.transform(Y_test.reshape(-1, 1)).T
    
    return X_train, Y_train, X_test, Y_test

# Function to select 100 images per class
def select_subset(X, Y, samples_per_class=100):
    classes = np.unique(np.argmax(Y, axis=0))  # Get unique classes (0-9)
    selected_indices = []

    for label in classes:
        # Find indices of all samples with the current label
        label_indices = np.where(np.argmax(Y, axis=0) == label)[0]
        # Randomly select 'samples_per_class' indices from each label
        selected_indices.extend(np.random.choice(label_indices, samples_per_class, replace=False))

    # Extract the selected samples
    X_subset = X[:, selected_indices]
    Y_subset = Y[:, selected_indices]
    return X_subset, Y_subset

# Load and preprocess data
X_train, Y_train, X_test, Y_test = preprocess_data()

# Extract 100 images per class
X_train_subset, Y_train_subset = select_subset(X_train, Y_train, samples_per_class=100)

# Initialize neural network
layer_dims = [784, 64, 32, 10]  # Define the architecture (input layer -> hidden layers -> output layer)
nn = NeuralNetwork(layer_dims, learning_rate=0.1)

# Train on the extracted subset
costs = nn.train(X_train_subset, Y_train_subset, num_iterations=1000)

# Plot cost curve to see the learning progression
plt.plot(costs)
plt.xlabel("Iterations (per hundreds)")
plt.ylabel("Cost")
plt.title("Cost reduction over iterations")
plt.show()

# Test the model on the original test dataset
predictions = nn.predict(X_test)
accuracy = nn.compute_accuracy(predictions, Y_test)

print(f"Test Accuracy: {accuracy:.2f}%")