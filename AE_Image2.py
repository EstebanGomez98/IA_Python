from tkinter import Tk, filedialog
import tkinter as tk
import numpy as np
from PIL import Image

x = np.array([98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98])
x2 = np.array([13, 13, 13, 13, 13, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 255, 255, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 13, 13])

def obtener_primera_letra():
    global x
    x = select_and_resize_image()

def obtener_segunda_letra():
    global x2
    x2 = select_and_resize_image()

def select_and_resize_image():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select an image file")

    if file_path:
        try:
            original_image = Image.open(file_path)
            resized_image = original_image.resize((5, 7))
            resized_image = resized_image.convert("L")
            pixel_data = list(resized_image.getdata())
            root.destroy()
            return pixel_data
        except Exception as e:
            root.destroy()
            return np.zeros(35)
    else:
        print("No file selected.")

def ventana():
    ventana = Tk()
    ventana.title("Ejercicio IA, Letras A y E")

    boton_obtener_primera_letra = tk.Button(ventana, text="Obtener primera letra", command=obtener_primera_letra)
    boton_obtener_primera_letra.grid(row=7, column=0, columnspan=2, pady=10)

    boton_obtener_segunda_letra = tk.Button(ventana, text="Obtener segunda letra", command=obtener_segunda_letra)
    boton_obtener_segunda_letra.grid(row=7, column=4, columnspan=2, pady=10)

    ventana.mainloop()

if __name__ == "__main__":
    ventana()
    
    w = np.random.rand(36)
    xr = np.array([x, x2])
    yd = np.array([1, 0])

    alfa = 0.004
    max_epochs = 1000

    weight_history = np.zeros((max_epochs, 36))
    error_history = np.zeros((max_epochs, 2))

    for epoch in range(max_epochs):
        error_sum = np.zeros(2)
        for i in range(2):
            tsum = np.dot(w, xr[i, :])
            y0 = (tsum > 0).astype(int)
            error = yd[i] - y0
            delta_w = alfa * error * xr[i, :]
            w = w + delta_w
            error_sum += np.abs(error)

        weight_history[epoch, :] = w
        error_history[epoch, :] = error_sum

        if np.all(error_sum == 0):
            break

    print('Epochs:', epoch + 1)
    print('Final Weights:')
    print(w)

    letra = ''
    imgInput = select_and_resize_image()
    imgInput.insert(0, 1)
    comp = np.dot(imgInput, w)
    if comp > 0:
        letra = 'A'
    else:
        letra = 'E'

    print(letra)
