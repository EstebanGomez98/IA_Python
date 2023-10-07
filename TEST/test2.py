import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import numpy as np


def aplicar_filtro(matriz, kernel):
    resultado = np.dot(matriz, kernel)
    return resultado


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
            matriz_red = imagen_redimensionada.convert("L")

            # Obtener el nombre del archivo y la extensión
            # nombre_archivo, extension = os.path.splitext(os.path.basename(ruta_imagen))

            # Guardar la imagen redimensionada con el mismo nombre que la imagen original y extensión .png
            # ruta_guardar = nombre_archivo + '_redimensionada.png'
            # imagen_redimensionada.save(ruta_guardar)

            # resultado_label.config(text='Imagen redimensionada y guardada como ' + ruta_guardar)

        except Exception as e:
            resultado_label.config(text='Error: ' + str(e))

    # Aplicar los filtros k1 y k2
    resultado_k1 = aplicar_filtro(matriz_red, k1)
    resultado_k2 = aplicar_filtro(matriz_red, k2)
    print("\n", resultado_k1)
    print("\n", resultado_k2)


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

# Iniciar el bucle principal de tkinter
ventana.mainloop()
