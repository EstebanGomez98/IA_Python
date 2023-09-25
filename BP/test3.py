import numpy as np
import pandas as pd
import PySimpleGUI as sg

# Define the ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Load your data (replace the path with your data file)
excel_file = r"D:\Windows Lib\Documents\Repos\IA_Python\BP\data.xlsx"  # Replace with the correct path
df = pd.read_excel(excel_file)
df.iloc[:, 2] = df.iloc[:, 2].astype(int).astype(str).astype(int)
selected_columns = df.iloc[:, :3].to_numpy()
last_column = df.iloc[:, -1].to_numpy()
array_of_arrays = last_column.reshape(-1, 1)

# Normalize the input data
X = selected_columns

# Outputs
y = array_of_arrays

# Initialize the neural network parameters
input_size = 3
hidden_size = 40
output_size = 1

# Hyperparameters
learning_rate = 0.001
momentum = 0.09

# Initialize weights with random initialization close to zero
np.random.seed(1)
weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01

# Initialize momentum terms
momentum_input_hidden = np.zeros_like(weights_input_hidden)
momentum_hidden_output = np.zeros_like(weights_hidden_output)

# Training the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward propagation with ReLU activation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = relu(output_layer_input)

    # Calculate the loss
    error = y - output_layer_output

    # Backpropagation with ReLU derivative
    d_output = error * relu_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_input)

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

# Create a function to predict the output
def predict_output(input_data):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = relu(output_layer_input)
    return predicted_output

# Define the layout for the PySimpleGUI window
layout = [
    [sg.Text("Enter Input 1"), sg.InputText(key="input1")],
    [sg.Text("Enter Input 2"), sg.InputText(key="input2")],
    [sg.Text("Enter Input 3"), sg.InputText(key="input3")],
    [sg.Button("Predict"), sg.Button("Exit")],
    [sg.Text("", size=(20, 1), key="output")]
]

# Create the PySimpleGUI window
window = sg.Window("Neural Network Predictor", layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    elif event == "Predict":
        input_data = np.array([
            float(values["input1"]), float(values["input2"]), float(values["input3"])
        ])
        predicted_output = predict_output(input_data)
        window["output"].update(f"Predicted Output: {predicted_output}")

window.close()
