import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the sigmoid activation function and its derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Load your data (replace the path with your data file)
excel_file = r"D:\Biblioteca\Documentos\IA_Python\RNA\BP\data.xlsx"
df = pd.read_excel(excel_file)
df.iloc[:, 2] = df.iloc[:, 2].astype(int).astype(str).astype(int)
selected_columns = df.iloc[:, :3].to_numpy()
last_column = df.iloc[:, -1].to_numpy()
array_of_arrays = last_column.reshape(-1, 1)

# Normalize the input data
X = (selected_columns - selected_columns.mean(axis=0)) / \
    selected_columns.std(axis=0)

# Outputs
y = array_of_arrays

# Initialize the neural network parameters
input_size = 3
hidden_size = 40
output_size = 1

# Hyperparameters
learning_rate = 0.01  # Adjust the learning rate if needed
momentum = 0.09

# Initialize weights with random initialization
np.random.seed(1)
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)

# Initialize momentum terms
momentum_input_hidden = np.zeros_like(weights_input_hidden)
momentum_hidden_output = np.zeros_like(weights_hidden_output)
errores = []
# Training the neural network
num_epochs = 1000  # Adjust the number of epochs
for epoch in range(num_epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    error = y - output_layer_output
    mean_squared_error = 0.5 * np.mean(error**2)  # MSE loss
    errores.append(mean_squared_error)

    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * \
        sigmoid_derivative(hidden_layer_output)

    # Update weights and biases with momentum
    momentum_input_hidden = (
        momentum * momentum_input_hidden + learning_rate * X.T.dot(d_hidden_layer))
    weights_input_hidden += momentum_input_hidden

    momentum_hidden_output = (
        momentum * momentum_hidden_output + learning_rate * hidden_layer_output.T.dot(d_output))
    weights_hidden_output += momentum_hidden_output

    # Calculate and print the loss on the training set periodically
    if epoch % 100 == 0:
        training_loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Training Loss: {training_loss:.4f}")

# Testing the neural network
for i in range(10):
    new_input = X[i, :]
    hidden_layer_input = np.dot(new_input, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    print(f"Input {i + 1}: {new_input}")
    print("-----")
    print(f"Desired output {i + 1}: {y[i, :]}")
    print("-----")
    print(f"Predicted output {i + 1}: {predicted_output}")

# Graficar el error cuadrático medio a lo largo de las épocas de entrenamiento
plt.plot(errores)
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio')
plt.title('Error de Entrenamiento (MSE) a lo largo de las Épocas')
plt.show()
