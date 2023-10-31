
import numpy as np
from tkinter import filedialog
from tkinter import Tk
from PIL import Image
import os


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

root = Tk()
root.withdraw()  # Ocultar la ventana principal

file_path = filedialog.askopenfilename(title="Selecciona una imagen",
                                       filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*")))

if not file_path:
    print('No se seleccionó ninguna imagen.')
else:
    try:
        imagen = Image.open(file_path)
        matriz_img = np.array(imagen.convert("L"))

        print('Proceso de convoluciones')
        resultado = convolucion(matriz_img, k5)
        resultado = convolucion(resultado, k3)
        resultado = convolucion(resultado, k4)
        print('Finalización Proceso de convoluciones')

        imagen_modificada = Image.fromarray(resultado.astype('uint8'))
        imagen_redimensionada = imagen_modificada.resize((100, 100))

        # Obtener el nombre del archivo sin la extensión
        nombre, extension = os.path.splitext(os.path.basename(file_path))
        # Crear el nuevo nombre del archivo con sufijo "_modificado"
        nuevo_nombre = f"{nombre}_modificado{extension}"

        # Guardar la imagen redimensionada con el nuevo nombre
        ruta_guardado = os.path.join(os.path.dirname(file_path), nuevo_nombre)
        imagen_redimensionada.save(ruta_guardado)

        print(f'Imagen guardada como: {ruta_guardado}')

    except Exception as e:
        print('Error: ', str(e))
