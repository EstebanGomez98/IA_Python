import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from imgMan import *

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
            imagen_redimensionada = imagen_modificada.resize((100, 100))

            # Mostrar la imagen modificada después de aplicar los filtros
            mostrar_imagen_modificada(imagen_redimensionada)

            # Enable the guardar_boton button
            guardar_imagen_modificada.config(
                state=tk.NORMAL, command=lambda: guardar_imagen_modificada(imagen_redimensionada))

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))
            print(e)

# Function to train
def entrenar():
    # Add your training code here
    resultado_label.config(text='Entrenamiento en progreso...')

# Function to recognize image
def reconocer_imagen():
    # Add your image recognition code here
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
