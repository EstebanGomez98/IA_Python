import numpy as np
import matplotlib.pyplot as plt

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
              [12, 2, 6, 0, 7],
              [5, 5, 4, 9, 5],
              [5, 8, 0, 3, 5],
              [16, 3, 7, 9, 1],
              [21, 12, 3, 8, 8],
              [14, 2, 4, 5, 9],
              [27, 5, 9, 0, 4],
              [27, 2, 6, 3, 3],
              [27, 11, 0, 2, 5],
              [4, 12, 8, 9, 1],
              [10, 1, 9, 4, 3],
              [22, 3, 3, 1, 4]])

# Define the corresponding labels
yd = np.array([[0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
              [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
              [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
              [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5],
              [0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
              [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5],
              [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
              [0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
              [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
              [0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Inicializar los parámetros de la red neuronal
input_size = 5
hidden_size1 = 20  # Aumentar el tamaño de la capa oculta
hidden_size2 = 20  # Aumentar el tamaño de la capa oculta
output_size = 10

# Alpha (tasa de aprendizaje)
learning_rate = 0.01  # Ajustar la tasa de aprendizaje

# Beta (momentum)
beta = 0.9  # Ajustar el valor de momentum

# Error deseado mínimo
error_deseado = 0.001

# Inicializar los pesos y sesgos para la primera capa oculta
np.random.seed(1)
weights_input_hidden1 = np.random.uniform(size=(input_size, hidden_size1))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden_size1, hidden_size2))

# Inicializar los pesos y sesgos para la segunda capa oculta
weights_hidden2_output = np.random.uniform(size=(hidden_size2, output_size))

# Inicializar los términos de momentum para la primera capa oculta
momentum_input_hidden1 = np.zeros_like(weights_input_hidden1)
momentum_hidden1_hidden2 = np.zeros_like(weights_hidden1_hidden2)

# Inicializar los términos de momentum para la segunda capa oculta
momentum_hidden2_output = np.zeros_like(weights_hidden2_output)

errores = []

# Entrenar la red neuronal
for epoch in range(100000):
    # Propagación hacia adelante
    hidden_layer1_input = np.dot(X, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calcular la pérdida (MSE)
    error = yd - output_layer_output
    mean_squared_error = 0.5 * np.mean(error**2)  # Pérdida MSE
    errores.append(mean_squared_error)

    # Comprobar si se alcanzó el error deseado mínimo
    if mean_squared_error <= error_deseado:
        print(
            f"Error mínimo deseado alcanzado en la época {epoch}. Deteniendo el entrenamiento.")
        break

    # Retropropagación
    d_output = error * sigmoid_derivative(output_layer_output)

    error_hidden_layer2 = d_output.dot(weights_hidden2_output.T)
    d_hidden_layer2 = error_hidden_layer2 * \
        sigmoid_derivative(hidden_layer2_output)

    error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
    d_hidden_layer1 = error_hidden_layer1 * \
        sigmoid_derivative(hidden_layer1_output)

    # Actualizar los pesos y sesgos con momentum para la segunda capa oculta
    momentum_hidden2_output = (beta * momentum_hidden2_output +
                               learning_rate * hidden_layer2_output.T.dot(d_output))
    weights_hidden2_output += momentum_hidden2_output

    # Actualizar los pesos y sesgos con momentum para la primera capa oculta
    momentum_hidden1_hidden2 = (beta * momentum_hidden1_hidden2 +
                                learning_rate * hidden_layer1_output.T.dot(d_hidden_layer2))
    weights_hidden1_hidden2 += momentum_hidden1_hidden2

    momentum_input_hidden1 = (beta * momentum_input_hidden1 +
                              learning_rate * X.T.dot(d_hidden_layer1))
    weights_input_hidden1 += momentum_input_hidden1

# Probar la red neuronal
new_input = np.array([17, 11, 8, 0, 3])
hidden_layer1_input = np.dot(new_input, weights_input_hidden1)
hidden_layer1_output = sigmoid(hidden_layer1_input)

hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = sigmoid(hidden_layer2_input)

output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
predicted_output = sigmoid(output_layer_input)
valores_redondeados = [round(valor, 1) for valor in predicted_output]

print("Entrada:", new_input)
print("Salida Predicha:", valores_redondeados)

# Graficar el error cuadrático medio a lo largo de las épocas de entrenamiento
plt.plot(errores)
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio')
plt.title('Error de Entrenamiento (MSE) a lo largo de las Épocas')
plt.show()
