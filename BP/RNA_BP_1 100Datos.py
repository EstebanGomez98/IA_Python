import numpy as np
import pandas as pd

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the training data

# Read the Excel file into a pandas DataFrame
excel_file = r"C:\Users\stran\OneDrive\Documents\Repos\IA_Python\BP\data.xlsx"  # Replace with the correct path
df = pd.read_excel(excel_file)
df.iloc[:, 2] = df.iloc[:, 2].astype(int).astype(str).astype(int)
selected_columns = df.iloc[:, :3].to_numpy()
last_column = df.iloc[:, -1].to_numpy()
array_of_arrays = last_column.reshape(-1, 1)
# Entradas
X = selected_columns
print("Entradas",selected_columns)

# Salidas
y = array_of_arrays  # Use the reshaped array as the target variable
print(array_of_arrays)
# Initialize the neural network parameters
input_size = 3
hidden_size = 20
output_size = 1

# Alpha
learning_rate = 0.01  # Adjust the learning rate if needed

# Beta (momentum)
beta = 0.09

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
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases with momentum
    momentum_input_hidden = (
        beta * momentum_input_hidden + learning_rate * X.T.dot(d_hidden_layer))
    weights_input_hidden += momentum_input_hidden

    momentum_hidden_output = (beta * momentum_hidden_output +
                              learning_rate * hidden_layer_output.T.dot(d_output))
    weights_hidden_output += momentum_hidden_output

# Testing the neural network
new_input = np.array([0.125693786272204, 0.261253789224246, 1.0])
hidden_layer_input = np.dot(new_input, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

print("Input:", new_input)
print("Predicted Output:", predicted_output)
