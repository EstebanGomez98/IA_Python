import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os
from imgMan import *
from ia_logica import *
# Define your convolucion and cambiar_tamano_array functions here

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

def mostrar_imagen_modificada(imagen):
    # Convert the modified image to RGB mode
    imagen_rgb = imagen.convert("RGB")

    # Convert the RGB image to PhotoImage format for Tkinter
    imagen_tk = ImageTk.PhotoImage(imagen_rgb)

    # Update the label to display the modified image
    imagen_label.config(image=imagen_tk)
    imagen_label.image = imagen_tk

def cambiar_tamano_imagen():
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

    # Solicitar al usuario que seleccione una imagen
    ruta_imagen = filedialog.askopenfilename()

    if ruta_imagen:
        try:
            # Abrir la imagen
            imagen = Image.open(ruta_imagen)

            # Cambiar el tamaño de la imagen
            # imagen_redimensionada = imagen.resize((800, 600))

            # Obtener la matriz de la imagen redimensionada
            matriz_img = np.array(imagen.convert("L"))

            # Aplicar los filtros k1 y k2
            print("inicando convoluciones con los filtros")
            print("convolucion 1")
            resultado = convolucion(matriz_img, k5)
            print("convolucion 2")
            # resultado = convolucion(resultado, k2)
            resultado = convolucion(resultado, k3)
            print("convolucion 3")
            resultado = convolucion(resultado, k4)
            print("terminando las convoluciones")
            # resultado = convolucion(resultado, k5)

            # Convert the result back to an Image object
            imagen_modificada = Image.fromarray(resultado.astype('uint8'))

            # Cambiar el tamaño de la imagen
            imagen_redimensionada = imagen_modificada.resize((20, 20))

            # Mostrar la imagen modificada después de aplicar los filtros
            mostrar_imagen_modificada(imagen_redimensionada)

            # Enable the guardar_boton button
            guardar_imagen_modificada.config(
                state=tk.NORMAL, command=lambda: guardar_imagen_modificada(imagen_redimensionada))

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))
            print(e)

# Function to train
# Function to train
def entrenar():
    # Define the input folder containing images and the output folder for resized images
    input_folder = "input_images/"
    output_folder = "output_images/"

    # Ensure the output folder exists, create it if necessary
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        resultado_label.config(text='No images found in the input folder.')
        return

    # Initialize empty lists to store input data (X) and labels (yd)
    X_list = []
    yd_list = []

    # Process all images from the input folder
    for i, image_filename in enumerate(image_files):
        try:
            # Construct the full path for the current image
            image_path = os.path.join(input_folder, image_filename)

            # Open the image
            imagen = Image.open(image_path)

            # Convert the image to an array
            matriz_img = np.array(imagen.convert("L"))

            # Apply transformations using cambiar_tamano_imagen
            resultado = cambiar_tamano_imagen(matriz_img)

            # Append the image array to the input data list (X)
            X_list.append(resultado)

            # Create a label (yd) for the current image (you need to define this logic)
            # For example, if you have labels for your images, you can create yd accordingly.
            # yd_list.append(label_for_current_image)

            # Save the modified image to the output folder
            output_filename = os.path.join(output_folder, f"image_{i + 1}_modified.png")
            imagen_modificada = Image.fromarray(resultado.astype('uint8'))
            imagen_modificada.save(output_filename)

        except Exception as e:
            print(f'Error processing image {image_filename}: {str(e)}')

    # Convert the input data lists to NumPy arrays
    X = np.array(X_list)
    yd = np.array(yd_list)

    # Call the train_neural_network function with X and yd
    errores = train_neural_network(X, yd, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, alpha, beta, error_deseado)

    resultado_label.config(text=f'Entrenamiento completo. {len(image_files)} imágenes procesadas y guardadas en la carpeta de salida.')


# Function to recognize image
def reconocer_imagen():
    # Call your recognition logic here
    resultado_label.config(text='Reconociendo imagen...')

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

reconocer_boton = tk.Button(ventana, text='Reconocer Imagen', command=reconocer_imagen)
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
