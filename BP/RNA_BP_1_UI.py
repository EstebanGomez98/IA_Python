import numpy as np
import PySimpleGUI as sg

# Define the sigmoid activation function and its derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the training data
X = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
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
hidden_size = 10
output_size = 1

# Alpha
learning_rate = 0.01

# Beta
beta = 0.10

# Initialize weights and biases
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Initialize momentum terms
momentum_input_hidden = np.zeros_like(weights_input_hidden)
momentum_hidden_output = np.zeros_like(weights_hidden_output)

# Training the neural network
for _ in range(100000):
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


# Define the layout of the GUI
layout = [
    [sg.Text("Ingrese los valores (separados por coma):")],
    [sg.InputText(key="-INPUT-")],
    [sg.Button("Predecir"), sg.Button("Salir")],
    [sg.Text("Valor predecido:"), sg.Text("", size=(15, 1), key="-OUTPUT-")],
]

window = sg.Window("Back Propagation", layout)

# Training the neural network (you can add your training code here)

while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, "Salir"):
        break

    if event == "Predecir":
        # Parse the input and convert it to a NumPy array
        new_input = np.array([float(x.strip())
                             for x in values["-INPUT-"].split(",")])

        # Forward propagation
        hidden_layer_input = np.dot(new_input, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)
        res = [round(valor, 0) for valor in predicted_output]
        print(res)

        # Update the GUI with the predicted output
        window["-OUTPUT-"].update(res)

window.close()
