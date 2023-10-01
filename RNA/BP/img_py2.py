from PIL import Image
import numpy as np

# Abre el archivo de imagen
image = Image.open(
    'D:\Biblioteca\Documentos\IA_Python\RNA\BP\imagenes\rgb_ia.png')

# Convierte la imagen a una matriz de NumPy
image_array = np.array(image)

# Divide la imagen en canales RGB
r, g, b = image.split()

# Convierte cada canal en una matriz NumPy
r_array = np.array(r)
g_array = np.array(g)
b_array = np.array(b)

# Imprime las matrices de los canales R, G y B
print("R", r_array)
print("G", g_array)
print("B", b_array)

# Muestra cada canal por separado
r.show(title='Red Channel')
g.show(title='Green Channel')
b.show(title='Blue Channel')

# Combina los canales en una matriz tridimensional (RGB)
rgb_channels = np.dstack((r_array, g_array, b_array))

# Calcula la diferencia de intensidad entre los canales R, G y B
intensity_diff = np.max(rgb_channels, axis=2) - np.min(rgb_channels, axis=2)

# Cierra el archivo de imagen cuando hayas terminado con él (opcional pero recomendado)
image.close()
