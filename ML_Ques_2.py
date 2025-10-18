from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

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


X_train, Y_train, X_test, Y_test = preprocess_data()
X_train_subset, Y_train_subset = select_subset(X_train, Y_train, samples_per_class=100)

pairs = []
pair_labels = []

# Create pairs of images
for i in range(X_train_subset.shape[1]):  # Iterate over the number of samples in X_train_subset
    for j in range(i + 1, X_train_subset.shape[1]):  # Ensure j > i, so we avoid duplicating pairs
        pairs.append((X_train_subset[:, i], X_train_subset[:, j]))  # Access samples using column indexing
        
        # Assign a label: 1 if the same class, 0 if different classes
        if np.array_equal(Y_train_subset[:, i], Y_train_subset[:, j]):  # Access labels using column indexing
            pair_labels.append(1)  # Same class
        else:
            pair_labels.append(0)  # Different classes

# Convert pairs to numpy arrays for further processing
pairs = np.array(pairs)
pair_labels = np.array(pair_labels)

# Example: Print the number of pairs created
print(f'Number of pairs created: {len(pairs)}')  # The number of pairs should be (N choose 2), i.e., (n*(n-1))/2
print(f'Pair labels shape: {pair_labels.shape}')  # Should match the number of pairs

import numpy as np

# Define activation functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

# Derivatives for backpropagation
def sigmoid_derivative(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)

def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return parameters

# Forward propagation
def forward_propagation(X, parameters):
    caches = {}
    A = X
    L = len(parameters) // 2

    for l in range(1, L + 1):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = np.dot(W, A_prev) + b
        A = relu(Z) if l < L else sigmoid(Z)  # ReLU for hidden layers, Sigmoid for output layer
        caches[f'A{l-1}'] = A_prev
        caches[f'Z{l}'] = Z

    return A, caches

# Compute binary cross-entropy cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
    return np.squeeze(cost)

# Backward propagation
def backward_propagation(AL, Y, parameters, caches):
    grads = {}
    L = len(parameters) // 2
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Derivative for last layer (sigmoid)
    dZ = AL - Y
    for l in reversed(range(1, L + 1)):
        A_prev = caches[f'A{l-1}']
        W = parameters[f'W{l}']
        Z = caches[f'Z{l}']

        grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
        grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 1:  # ReLU for hidden layers
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * relu_derivative(caches[f'Z{l-1}'])

    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters

# Create mini-batches
def create_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    mini_batches = []
    
    # Shuffle the data
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    
    # Split the data into mini-batches
    num_batches = m // batch_size
    for i in range(num_batches):
        X_batch = X_shuffled[:, i*batch_size : (i+1)*batch_size]
        Y_batch = Y_shuffled[:, i*batch_size : (i+1)*batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    # Handle last batch if not an exact division
    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_batches*batch_size:]
        Y_batch = Y_shuffled[:, num_batches*batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches

# Model training function with mini-batches
def train(X, Y, layer_dims, learning_rate=0.01, num_iterations=1000, batch_size=64):
    parameters = initialize_parameters(layer_dims)
    costs = []

    for i in range(num_iterations):
        mini_batches = create_mini_batches(X, Y, batch_size)
        
        # Loop over mini-batches
        for X_batch, Y_batch in mini_batches:
            AL, caches = forward_propagation(X_batch, parameters)
            cost = compute_cost(AL, Y_batch)
            grads = backward_propagation(AL, Y_batch, parameters, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 10 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    return parameters, costs

# Predict function
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

# Accuracy calculation
def compute_accuracy(predictions, Y):
    return np.mean(predictions == Y) * 100

# Training on the created pairs
# Assuming pairs is of shape (2, num_pairs, image_flat_size) and pair_labels is the binary labels array


# Testing and accuracy computation can be done after training
pairs_flat = np.array([np.concatenate((x[0], x[1])) for x in pairs]).T  # Flatten pairs for input
pair_labels_flat = pair_labels.reshape(1, -1)

# Define MLP layer dimensions
layer_dims = [pairs_flat.shape[0], 64, 1]  # Input, hidden layers, output layer

# Train the model with mini-batches
parameters, costs = train(pairs_flat, pair_labels_flat, layer_dims, learning_rate=0.01, num_iterations=100, batch_size=800)
# Testing

def store_k_images_per_class(X_train_subset, Y_train_subset, k):
    """
    Store k random images for each class in a dictionary.
    """
    class_images = {}
    
    for class_label in range(10):  # Assuming 10 classes (0-9)
        # Find indices of all images for the current class
        class_indices = np.where(Y_train_subset[class_label] == 1)[0]
        # Randomly select k images from this class
        chosen_indices = np.random.choice(class_indices, k)
        # Store selected images for this class
        class_images[class_label] = X_train_subset[:, chosen_indices]
    
    return class_images

# Example setup
k_values = range(11, 30)  # Number of images per class to store
class_images = store_k_images_per_class(X_train_subset, Y_train_subset, k_values)

def create_test_pairs_with_stored_images(X_test, class_images):
    """
    Create pairs of each test image with the k stored images for each class.
    """
    test_pairs = []
    
    for i, test_image in enumerate(X_test.T):  # Iterate over each test sample
        class_pairs = []
        for class_label, images in class_images.items():
            for j in range(images.shape[1]):
                class_pairs.append((test_image, images[:, j]))  # Pair test image with each stored training image
        test_pairs.append(class_pairs)
    
    return np.array(test_pairs)

# Generate pairs for test images
accuracy=[]
for k in k_values:
    class_images = store_k_images_per_class(X_train_subset, Y_train_subset, k)
    test_pairs = create_test_pairs_with_stored_images(X_test, class_images)
    
    # Flatten each pair for model input
    test_pairs_flat = np.array([np.concatenate((pair[0], pair[1])) for class_pairs in test_pairs for pair in class_pairs])
    
    # Run predictions on all test pairs
    test_predictions = predict(test_pairs_flat.T, parameters)  # .T to match input shape for predict function
    
    # Reshape predictions to get counts for each test image
    test_predictions = test_predictions.reshape(len(X_test.T), 10, k)  # Reshape for test images, classes, and pairs
    
    # Determine the predicted class by counting matches
    predicted_labels = []
    for i in range(test_predictions.shape[0]):  # For each test image
        match_counts = np.sum(test_predictions[i], axis=1)  # Count matches per class
        predicted_class = np.argmax(match_counts)  # Class with the highest match count
        predicted_labels.append(predicted_class)
    
    predicted_labels = np.array(predicted_labels)
    
    # Calculate accuracy
    true_labels = np.argmax(Y_test, axis=0)  # Assuming one-hot encoded test labels
    accuracy = np.mean(predicted_labels == true_labels) * 100
    print(f"Model accuracy on test data: {accuracy:.2f}%")
