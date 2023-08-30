import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Configuración de la RNA
input_size = 35
hidden_size = 10
output_size = 3
learning_rate = 0.1
epochs = 10000

# Datos de entrenamiento
# Aquí debes proporcionar tus propios datos de entrenamiento
# Cada fila representa una entrada de 35 dimensiones y su correspondiente salida de 3 dimensiones.
# Asegúrate de que los datos estén en formato numpy array.
X = np.array([
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
])

Y = np.array([
    # Salida correspondiente al Patrón 1
    [1, 0, 0],
    # Salida correspondiente al Patrón 2
    [0, 1, 0],
    # Salida correspondiente al Patrón 3
    [0, 0, 1],
    # Salida correspondiente al Patrón 4
    [1, 1, 0],
    # Salida correspondiente al Patrón 5
    [0, 1, 1],
])

# Inicialización de los pesos de forma aleatoria
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Entrenamiento de la RNA
for epoch in range(epochs):
    # Propagación hacia adelante
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Cálculo del error
    error = Y - output_layer_output

    # Retropropagación del error
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Actualización de los pesos
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    # Verifique la convergencia (el error es cero)
    if np.all(d_output == 0):
        print("finaliza el entrenamientoA")
        break
print("finaliza el entrenamientoB")

# Predicción
new_input = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])  # Reemplaza con tus datos de entrada
hidden_layer_input = np.dot(new_input, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

valores_redondeados = [round(valor, 0) for valor in predicted_output]

# Imprimir los valores redondeados
print(valores_redondeados)