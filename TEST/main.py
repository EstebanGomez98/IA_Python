import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
from imgMan import *
from ia_logica import *


def guardar_imagen_modificada(imagen):
    # Ask the user for the save file location
    ruta_guardar = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG files", "*.png")])

    if ruta_guardar:
        try:
            # Save the modified image
            imagen.save(ruta_guardar)
            resultado_label.config(text='Imagen guardada como ' + ruta_guardar)
        except Exception as e:
            resultado_label.config(
                text='Error al guardar la imagen: ' + str(e))

def convolucion(imagen, kernel):
    altura_imagen, ancho_imagen = len(imagen), len(imagen[0])
    altura_kernel, ancho_kernel = len(kernel), len(kernel[0])
    resultado = np.zeros((altura_imagen - altura_kernel + 1,
    ancho_imagen - ancho_kernel + 1), dtype=np.float32)
    # Iterar a través de la imagen
    for i in range(altura_imagen - altura_kernel + 1):
        for j in range(ancho_imagen - ancho_kernel + 1):
            suma = 0
            # Realizar la convolución en la región de la imagen correspondiente al tamaño del kernel
            for x in range(altura_kernel):
                for y in range(ancho_kernel):
                    suma += imagen[i + x][j + y] * kernel[x][y]
            resultado[i][j] = suma
    return resultado

def mostrar_imagen_modificada(imagen):
    # Convert the modified image to RGB mode
    imagen_rgb = imagen.convert("RGB")

    # Convert the RGB image to PhotoImage format for Tkinter
    imagen_tk = ImageTk.PhotoImage(imagen_rgb)

    # Update the label to display the modified image
    imagen_label.config(image=imagen_tk)
    imagen_label.image = imagen_tk

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

    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, error


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

def cambiar_tamano_imagen(imagen):
    output_folder = "output_images/"

    os.makedirs(output_folder, exist_ok=True)

    k1 = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]])

    k2 = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]])

    # enfoque
    k3 = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])

    # shapen
    k4 = np.array([
        [1, -2, 1],
        [-2, 5, -2],
        [1, -2, 1]])

    # deteccion de bordes
    k5 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])

    if imagen:
        try:
            # Convert the image to an array
            matriz_img = np.array(imagen.convert("L"))
            print("convolucion 1")
            resultado = convolucion(matriz_img, k5)
            print("convolucion 2")
            resultado = convolucion(resultado, k3)
            print("convolucion 3")
            resultado = convolucion(resultado, k4)
            print("fin convolucion")
            imagen_modificada = Image.fromarray(resultado.astype('uint8'))
            imagen_redimensionada = imagen_modificada.resize((20, 20))
            pixel_data = list(imagen_redimensionada.getdata())

            return pixel_data

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))


def entrenar():

    global wih1, wh1h2, wh2o, error

    # Define your input data and labels
    X = np.zeros((1, 400))
    yd = np.zeros((1, 3))

    input_size = 400
    hidden_size1 = 100
    hidden_size2 = 100
    output_size = 3

    alpha = 0.001
    beta = 0.9
    error_deseado = 0.01

    input_folder = "input_images/"

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(
        input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        resultado_label.config(text='No images found in the input folder.')
        return

    # Process all images from the input folder
    for i, image_filename in enumerate(image_files):
        try:
            # Construct the full path for the current image
            image_path = os.path.join(input_folder, image_filename)

            # Open the image
            imagen = Image.open(image_path)

            print(image_filename)
            # Apply transformations using cambiar_tamano_imagen
            resultado = cambiar_tamano_imagen(imagen)
            X = np.vstack((X, resultado))

        except Exception as e:
            print(f'Error processing image {image_filename}: {str(e)}')

    # Convert the input data lists to NumPy arrays
    # A
    for i in range(5):
        yd = np.vstack((yd, [1.0, 0.0, 0.0]))
    # E
    # for i in range(5):
    #    yd = np.vstack((yd, [0.0, 1.0, 0.0]))
    # I
    # for i in range(5):
    #    yd = np.vstack((yd, [0.0, 0.0, 1.0]))
    # O
    # for i in range(5):
    #    yd = np.vstack((yd, [1.0, 1.0, 0.0]))
    # U
    # for i in range(5):
    #    yd = np.vstack((yd, [0.0, 1.0, 1.0]))

    resultado_label.config(text='inicializando pesos')
    weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = initialize_weights(
        input_size, hidden_size1, hidden_size2, output_size)

    resultado_label.config(text='Entrenamiento en progreso...')
    wih1, wh1h2, wh2o, error = train_neural_network(
        X, yd, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, alpha, beta, error_deseado)

    resultado_label.config(text='Fin Entrenamiento')

# Function to recognize image


def reconocer_imagen():
    # Ask the user to select an image
    ruta_imagen = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")])

    if ruta_imagen:
        try:
            # Call the cambiar_tamano_imagen function with the selected image
            new_input = cambiar_tamano_imagen(ruta_imagen)

            if new_input is not None:
                # Use the new_input for recognition
                valores_redondeados = test_neural_network(
                    new_input, wih1, wh1h2, wh2o)
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
                    case _:
                        letter = "No se que hacer"

                # Update the result label with the recognized letter
                resultado_label.config(text=letter)
            else:
                resultado_label.config(text='Error processing the image.')

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))


# Function to exit the program


def salir():
    ventana.quit()


# Create a tkinter window with custom dimensions
ventana = tk.Tk()
ventana.title('Entrenamiento y Reconocimiento de Imágenes')

# Define the dimensions of the window (width x height)
dimensiones_ventana = "800x600"
ventana.geometry(dimensiones_ventana)

# Buttons
entrenar_boton = tk.Button(ventana, text='Entrenar', command=entrenar)
entrenar_boton.pack(pady=50)

reconocer_boton = tk.Button(
    ventana, text='Reconocer Imagen', command=reconocer_imagen)
reconocer_boton.pack()

salir_boton = tk.Button(ventana, text='Salir', command=salir)
salir_boton.pack()

# Label to display the result
resultado_label = tk.Label(ventana, text='')
resultado_label.pack()

# Label to display the modified image
imagen_label = tk.Label(ventana)
imagen_label.pack()

# Start the main tkinter loop
ventana.mainloop()
