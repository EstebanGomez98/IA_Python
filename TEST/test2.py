import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

def aplicar_filtro(matriz, kernel):
    m, n = matriz.shape
    kaltura, kancho = kernel.shape
    kaltura_half, kancho_half = kaltura // 2, kancho // 2
    
    resultado = np.zeros((m, n), dtype=np.float32)
    
    for i in range(kaltura_half, m - kaltura_half):
        for j in range(kancho_half, n - kancho_half):
            suma = 0
            for x in range(kaltura):
                for y in range(kancho):
                    suma += matriz[i - kaltura_half + x, j - kancho_half + y] * kernel[x, y]
            resultado[i, j] = suma
    
    return resultado

def mostrar_imagen_modificada(imagen):
    # Convert the modified image to RGB mode
    imagen_rgb = imagen.convert("RGB")
    
    # Convert the RGB image to PhotoImage format for Tkinter
    imagen_tk = ImageTk.PhotoImage(imagen_rgb)

    # Update the label to display the modified image
    imagen_label.config(image=imagen_tk)
    imagen_label.image = imagen_tk

def guardar_imagen_modificada(imagen):
    # Ask the user for the save file location
    ruta_guardar = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])

    if ruta_guardar:
        try:
            # Save the modified image
            imagen.save(ruta_guardar)
            resultado_label.config(text='Imagen guardada como ' + ruta_guardar)
        except Exception as e:
            resultado_label.config(text='Error al guardar la imagen: ' + str(e))

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
            resultado_k2 = aplicar_filtro(resultado_k1, k2)

            # Convert the result back to an Image object
            imagen_modificada = Image.fromarray(resultado_k2.astype('uint8'))

            # Mostrar la imagen modificada después de aplicar los filtros
            mostrar_imagen_modificada(imagen_modificada)

            # Enable the guardar_boton button
            guardar_boton.config(state=tk.NORMAL, command=lambda: guardar_imagen_modificada(imagen_modificada))

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))

# Crear una ventana de tkinter
ventana = tk.Tk()
ventana.title('Redimensionar y Filtrar Imagen')

# Botón para seleccionar una imagen
seleccionar_boton = tk.Button(
    ventana, text='Seleccionar Imagen', command=cambiar_tamano_imagen)
seleccionar_boton.pack(pady=20)

# Botón para guardar la imagen modificada
guardar_boton = tk.Button(
    ventana, text='Guardar Imagen Modificada', state=tk.DISABLED)
guardar_boton.pack()

# Etiqueta para mostrar el resultado
resultado_label = tk.Label(ventana, text='')
resultado_label.pack()

# Etiqueta para mostrar la imagen redimensionada
imagen_label = tk.Label(ventana)
imagen_label.pack()

# Iniciar el bucle principal de tkinter
ventana.mainloop()
