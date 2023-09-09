import PySimpleGUI as sg
import numpy as np
from PIL import Image

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

def normalization(x):
    nueva_lista = [1 if valor / 255 == 0 else 0 for valor in x]
    return nueva_lista
# Define el tamaño de la fuente
font_size = 14  # Puedes ajustar el tamaño de la fuente aquí

# Define the layout for the GUI
layout = [
    [sg.Text("Seleccione las imagenes para el entrenamiento", font=("AnyFont", font_size))],
    [sg.Button("Train Network", font=("AnyFont", font_size)), sg.Button("Select Image", font=("AnyFont", font_size)), sg.Button("Exit", font=("AnyFont", font_size))],
    [sg.Text("", size=(30, 1), key="-OUTPUT-")],
]

# Create the window
window = sg.Window("RNA Vocales", layout)

# Training function
def train_network():
    window["Train Network"].update(disabled=True)
    global weights_input_hidden, weights_hidden_output, input, yd
    input = np.zeros((1, 35))
    yd = np.zeros((1, 3))
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

    #repetir las veces que quiera y dependiendo de la letra adicionele a la lista y 
    for i in range(3):
        sg.popup("Seleccione la imagen de la letra A", i+1)
        x = select_image()
        x = normalization(x)
        input = np.vstack((input, x))
        yd = np.vstack((yd, [1.0, 0.0, 0.0]))

    for i in range(3):
        sg.popup("Seleccione la imagen de la letra E", i+1)
        x2 = select_image()
        x2 = normalization(x2)
        input = np.vstack((input, x2))
        yd = np.vstack((yd, [0.0, 1.0, 0.0]))

    for i in range(3):
        sg.popup("Seleccione la imagen de la letra I", i+1)
        x3 = select_image()
        x3 = normalization(x3)
        input = np.vstack((input, x3))
        yd = np.vstack((yd, [0.0, 0.0, 1.0]))

    for i in range(3):
        sg.popup("Seleccione la imagen de la letra O", i+1)
        x4 = select_image()
        x4 = normalization(x4)
        input = np.vstack((input, x4))
        yd = np.vstack((yd, [1.0, 1.0, 0.0]))

    for i in range(3):
        sg.popup("Seleccione la imagen de la letra U", i+1)
        x5 = select_image()
        x5 = normalization(x5)
        input = np.vstack((input, x5))
        yd = np.vstack((yd, [0.0, 1.0, 1.0]))

    input = input[1:]
    yd = yd[1:]

    # Inicialización de los pesos de forma aleatoria
    np.random.seed(1)
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    global epoch
    # Entrenamiento de la RNA
    for epoch in range(epochs):
        # Propagación hacia adelante
        hidden_layer_input = np.dot(input, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)

        # Cálculo del error
        error = yd - output_layer_output

        # Retropropagación del error
        d_output = error * sigmoid_derivative(output_layer_output)
        error_hidden_layer = d_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Actualización de los pesos
        weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
        weights_input_hidden += input.T.dot(d_hidden_layer) * learning_rate

        # Verifique la convergencia (el error es cero)
        if np.all(d_output == 0):
            sg.popup("Entrenamiento completado")
            break
    sg.popup("Entrenamiento completado")

# Image selection function
def select_image():
    root = sg.tk.Tk()
    root.withdraw()

    file_path = sg.popup_get_file("Select an image file", file_types=(("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"),))

    if file_path:
        try:
            original_image = Image.open(file_path)
            resized_image = original_image.resize((5, 7))
            resized_image = resized_image.convert("L")
            pixel_data = list(resized_image.getdata())
            return pixel_data
        except Exception as e:
            sg.popup_error("Error processing the image.")
            return np.zeros(35)
    else:
        sg.popup("No file selected.")

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    elif event == "Train Network":
        train_network()

    elif event == "Select Image":
        new_input = select_image()
        new_input = normalization(new_input)
        if new_input:
            hidden_layer_input = np.dot(new_input, weights_input_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
            predicted_output = sigmoid(output_layer_input)
            valores_redondeados = [round(valor, 0) for valor in predicted_output]
            letter = ""
            match valores_redondeados:
                case [1.0, 0.0, 0.0]:
                    letter = "A"
                case [0.0, 1.0, 0.0]:
                    letter = "E"
                case [0.0, 0.0, 1.0]:
                    letter = "I"
                case [1.0, 1.0, 0.0]:
                    letter = "O"
                case [0.0, 1.0, 1.0]:
                    letter = "U"
            window["-OUTPUT-"].update(f"Predicted Letter: {letter}")
            # Imprimir los valores redondeados
            print(valores_redondeados)

# Close the window
window.close()
