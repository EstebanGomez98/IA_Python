import numpy as np

# Define the sigmoid activation function and its derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the training data
X = np.array([[17, 11, 8, 0, 3],
              [4, 11, 9, 8, 1],
              [30, 6, 4, 8, 5],
              [22, 8, 3, 2, 9],
              [13, 3, 6, 4, 6],
              [19, 11, 4, 7, 3],
              [17, 12, 7, 1, 8],
              [18, 5, 9, 5, 9],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]])

# Define the corresponding labels
y = np.array([[0],
              [1],
              [1],
              [0],
              [1],
              [0],
              [0],
              [1],
              [1],
              [0],
              [0],
              [1],
              [0],
              [1],
              [1],
              [0]])

# Initialize the neural network parameters
input_size = 4
hidden_size = 3
output_size = 10

# Alpha
learning_rate = 0.001

# Beta
beta = 0.9

# Initialize weights and biases
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Initialize momentum terms
momentum_input_hidden = np.zeros_like(weights_input_hidden)
momentum_hidden_output = np.zeros_like(weights_hidden_output)

# Training the neural network
for epoch in range(10000):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    error = y - output_layer_output

    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * \
        sigmoid_derivative(hidden_layer_output)

    # Update weights and biases with momentum
    momentum_input_hidden = (
        beta * momentum_input_hidden + learning_rate * X.T.dot(d_hidden_layer))
    weights_input_hidden += momentum_input_hidden

    momentum_hidden_output = (beta * momentum_hidden_output +
                              learning_rate * hidden_layer_output.T.dot(d_output))
    weights_hidden_output += momentum_hidden_output

# Testing the neural network
new_input = np.array([0, 1, 0, 0])
hidden_layer_input = np.dot(new_input, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

print("Input:", new_input)
print("Predicted Output:", predicted_output)
