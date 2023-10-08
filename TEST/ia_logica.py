import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    np.random.seed(1)
    weights_input_hidden1 = np.random.randn(input_size, hidden_size1)
    weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)
    weights_hidden2_output = np.random.randn(hidden_size2, output_size)
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output


def train_neural_network(X, yd, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, alpha, beta, error_deseado):
    momentum_input_hidden1 = np.zeros_like(weights_input_hidden1)
    momentum_hidden1_hidden2 = np.zeros_like(weights_hidden1_hidden2)
    momentum_hidden2_output = np.zeros_like(weights_hidden2_output)
    errores = []

    while True:
        # Forward propagation
        hidden_layer1_input = np.dot(X, weights_input_hidden1)
        hidden_layer1_output = sigmoid(hidden_layer1_input)

        hidden_layer2_input = np.dot(
            hidden_layer1_output, weights_hidden1_hidden2)
        hidden_layer2_output = sigmoid(hidden_layer2_input)

        output_layer_input = np.dot(
            hidden_layer2_output, weights_hidden2_output)
        output_layer_output = sigmoid(output_layer_input)

        # Calculate the loss
        error = yd - output_layer_output
        mean_squared_error = 0.5 * np.mean(error**2)
        errores.append(mean_squared_error)

        if mean_squared_error <= error_deseado:
            print(f"Error deseado alcanzado. Deteniendo el entrenamiento.")
            break

        # Backpropagation
        d_output = error * sigmoid_derivative(output_layer_output)

        error_hidden_layer2 = d_output.dot(weights_hidden2_output.T)
        d_hidden_layer2 = error_hidden_layer2 * \
            sigmoid_derivative(hidden_layer2_output)

        error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
        d_hidden_layer1 = error_hidden_layer1 * \
            sigmoid_derivative(hidden_layer1_output)

        # Weight updates with momentum
        momentum_hidden2_output = (beta * momentum_hidden2_output +
                                   alpha * hidden_layer2_output.T.dot(d_output))
        weights_hidden2_output += momentum_hidden2_output

        momentum_hidden1_hidden2 = (beta * momentum_hidden1_hidden2 +
                                    alpha * hidden_layer1_output.T.dot(d_hidden_layer2))
        weights_hidden1_hidden2 += momentum_hidden1_hidden2

        momentum_input_hidden1 = (beta * momentum_input_hidden1 +
                                  alpha * X.T.dot(d_hidden_layer1))
        weights_input_hidden1 += momentum_input_hidden1

    return errores


def test_neural_network(new_input, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output):
    # Forward propagation for testing
    hidden_layer1_input = np.dot(new_input, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
    predicted_output = sigmoid(output_layer_input)
    valores_redondeados = [round(valor, 0) for valor in predicted_output]

    return valores_redondeados
