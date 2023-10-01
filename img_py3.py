from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Abre el archivo de imagen
image = Image.open('PEZCIRUJANO.jpg')

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

# Crea una figura con 2 filas y 2 columnas para mostrar las imágenes
plt.figure(figsize=(12, 8))

# Muestra la imagen original en la primera subtrama
plt.subplot(2, 2, 1)
plt.imshow(image_array)
plt.title('Original Image')
plt.axis('off')

# Muestra el canal rojo en la segunda subtrama
plt.subplot(2, 2, 2)
plt.imshow(r_array, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

# Muestra el canal verde en la tercera subtrama
plt.subplot(2, 2, 3)
plt.imshow(g_array, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

# Muestra el canal azul en la cuarta subtrama
plt.subplot(2, 2, 4)
plt.imshow(b_array, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

# Muestra la ventana con todas las subtramas
plt.tight_layout()
plt.show()

# Cierra el archivo de imagen cuando hayas terminado con él (opcional pero recomendado)
image.close()
