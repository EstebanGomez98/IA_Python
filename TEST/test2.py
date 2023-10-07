import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np

def aplicar_filtro(matriz, kernel):
    altura, ancho = matriz.shape
    kaltura, kancho = kernel.shape
    resultado = np.zeros((altura - kaltura + 1, ancho - kancho + 1))
    for i in range(altura - kaltura + 1):
        for j in range(ancho - kancho + 1):
            resultado[i, j] = np.sum(matriz[i:i+kaltura, j:j+kancho] * kernel)
    return resultado

def mostrar_imagen_modificada(matriz):
    # Create a new Image object from the modified matrix
    imagen_modificada = Image.fromarray(matriz.astype('uint8'))

    # Convert the modified image to PhotoImage format for Tkinter
    imagen_tk = ImageTk.PhotoImage(imagen_modificada)

    # Update the label to display the modified image
    imagen_label.config(image=imagen_tk)
    imagen_label.image = imagen_tk

def cambiar_tamano_imagen():
    # Solicitar al usuario que seleccione una imagen
    ruta_imagen = filedialog.askopenfilename()

    k1 = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]])

    k2 = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]])

    global matriz_red

    if ruta_imagen:
        try:
            # Abrir la imagen
            imagen = Image.open(ruta_imagen)

            # Cambiar el tamaño de la imagen
            imagen_redimensionada = imagen.resize((24, 24))

            # Obtener la matriz de la imagen redimensionada
            matriz_red = np.array(imagen_redimensionada.convert("L"))

            # Aplicar los filtros k1 y k2
            resultado_k1 = aplicar_filtro(matriz_red, k1)
            resultado_k2 = aplicar_filtro(matriz_red, k2)

            # Mostrar la imagen modificada después de aplicar los filtros
            mostrar_imagen_modificada(resultado_k1)  # You can choose either k1 or k2 here.

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))

# Crear una ventana de tkinter
ventana = tk.Tk()
ventana.title('Redimensionar Imagen')

# Botón para seleccionar una imagen
seleccionar_boton = tk.Button(
    ventana, text='Seleccionar Imagen', command=cambiar_tamano_imagen)
seleccionar_boton.pack(pady=20)

# Etiqueta para mostrar el resultado
resultado_label = tk.Label(ventana, text='')
resultado_label.pack()

# Etiqueta para mostrar la imagen redimensionada
imagen_label = tk.Label(ventana)
imagen_label.pack()

# Iniciar el bucle principal de tkinter
ventana.mainloop()
