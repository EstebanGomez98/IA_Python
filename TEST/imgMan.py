from PIL import Image, ImageTk
import numpy as np

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

