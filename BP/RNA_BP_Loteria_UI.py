import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# Definir la función de activación sigmoide y su derivada
layout = [
    [sg.Text('Valores de entrada (separados por coma):')],
    [sg.InputText(key='-INPUT-', size=(20, 1))],
    [sg.Button('Predecir'), sg.Exit("Salir")],
    [sg.Text('Valor predecido:'), sg.Text('', size=(20, 1), key='-OUTPUT-')]
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Definir los datos de entrenamiento
X = np.array([[17.1, 11.1, 8.1, 0.1, 3.1],
              [4.1, 11.1, 9.1, 8.1, 1.1],
              [30.1, 6.1, 4.1, 8.1, 5.1],
              [22.1, 8.1, 3.1, 2.1, 9.1],
              [13.1, 3.1, 6.1, 4.1, 6.1],
              [19.1, 11.1, 4.1, 7.1, 3.1],
              [17.1, 12.1, 7.1, 1.1, 8.1],
              [18.1, 5.1, 9.1, 5.1, 9.1],
              [12.1, 2.1, 6.1, 0.1, 7.1],
              [5.1, 5.1, 4.1, 9.1, 5.1],
              [5.1, 8.1, 0.1, 3.1, 5.1],
              [16.1, 3.1, 7.1, 9.1, 1.1],
              [21.1, 12.1, 3.1, 8.1, 8.1],
              [14.1, 2.1, 4.1, 5.1, 9.1],
              [27.1, 5.1, 9.1, 0.1, 4.1],
              [27.1, 2.1, 6.1, 3.1, 3.1],
              [27.1, 11.1, 0.1, 2.1, 5.1],
              [4.1, 12.1, 8.1, 9.1, 1.1],
              [10.1, 1.1, 9.1, 4.1, 3.1],
              [22.1, 3.1, 3.1, 1.1, 4.1]])

# Definir las etiquetas correspondientes
yd = np.array([[0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 1.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1],
              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 1.1],
              [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5],
              [0.5, 0.1, 0.1, 0.5, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.5],
              [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 1.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5],
              [0.5, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.1, 0.1, 0.1, 1.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.5, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5],
              [0.1, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]])
print(yd)
# Initialize the neural network parameters
input_size = 5
hidden_size1 = 20  # Tamaño de la primera capa oculta
hidden_size2 = 15  # Tamaño de la segunda capa oculta
output_size = 10

# Alpha (tasa de aprendizaje)
alpha = 0.001

# Beta (momentum)
beta = 0.9

# Error deseado mínimo
error_deseado = 0.01

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

# Training the neural network
for epoch in range(10000):  # Puedes ajustar el número de épocas según sea necesario
    # Forward propagation
    hidden_layer1_input = np.dot(X, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    error = yd - output_layer_output
    mean_squared_error = 0.5 * np.mean(error**2)  # MSE loss
    errores.append(mean_squared_error)

    if mean_squared_error <= error_deseado:
        print(
            f"Error deseado alcanzado en la época {epoch}. Deteniendo el entrenamiento.")
        break

    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)

    error_hidden_layer2 = d_output.dot(weights_hidden2_output.T)
    d_hidden_layer2 = error_hidden_layer2 * \
        sigmoid_derivative(hidden_layer2_output)

    error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
    d_hidden_layer1 = error_hidden_layer1 * \
        sigmoid_derivative(hidden_layer1_output)

    # Actualice pesos y sesgos con momentum para la segunda capa oculta
    momentum_hidden2_output = (beta * momentum_hidden2_output +
                               alpha * hidden_layer2_output.T.dot(d_output))
    weights_hidden2_output += momentum_hidden2_output

    # Actualice pesos y sesgos con momentum para la primera capa oculta
    momentum_hidden1_hidden2 = (beta * momentum_hidden1_hidden2 +
                                alpha * hidden_layer1_output.T.dot(d_hidden_layer2))
    weights_hidden1_hidden2 += momentum_hidden1_hidden2

    momentum_input_hidden1 = (beta * momentum_input_hidden1 +
                              alpha * X.T.dot(d_hidden_layer1))
    weights_input_hidden1 += momentum_input_hidden1

# Testing the neural network
new_input = np.array([22.1, 3.1, 3.1, 1.1, 4.1])
hidden_layer1_input = np.dot(new_input, weights_input_hidden1)
hidden_layer1_output = sigmoid(hidden_layer1_input)

hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = sigmoid(hidden_layer2_input)

output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
predicted_output = sigmoid(output_layer_input)
valores_redondeados = [round(valor, 1) for valor in predicted_output]

window = sg.Window('RNA Back Propagation', layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Salir':
        break

    if event == 'Predecir':
        # Parse and normalize the input
        input_str = values['-INPUT-']
        input_values = [float(val) for val in input_str.split(',')]
        input_values = (input_values - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Feed forward through the neural network
        hidden_layer1_input = np.dot(input_values, weights_input_hidden1)
        hidden_layer1_output = sigmoid(hidden_layer1_input)

        hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
        hidden_layer2_output = sigmoid(hidden_layer2_input)

        output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
        predicted_output = sigmoid(output_layer_input)
        predicted_output_rounded = [round(val, 1) for val in predicted_output]

        # Update the output field in the GUI
        window['-OUTPUT-'].update(', '.join(map(str, predicted_output_rounded)))

window.close()

# Graficar el error cuadrático medio a lo largo de las épocas de entrenamiento
plt.plot(errores)
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio')
plt.title('Error de Entrenamiento (MSE) a lo largo de las Épocas')
plt.show()
