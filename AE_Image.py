
from tkinter import Tk,filedialog
import tkinter as tk
import numpy as np;
from PIL import Image

def select_and_resize_image():
    # Create a Tkinter root window (it will not be displayed)
    root = Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(title="Select an image file")

    if file_path:
        try:
            # Open the selected image using PIL
            original_image = Image.open(file_path)

            # Resize the image to 5x7 pixels
            resized_image = original_image.resize((5, 7))

            # Convert the resized image to grayscale
            resized_image = resized_image.convert("L")

            # Get the pixel data as a list
            pixel_data = list(resized_image.getdata())
            #resized_image.show()
            # Close the image
            resized_image.close()

            # Display the pixel data
            root.quit()
            return pixel_data

        except Exception as e:
            root.quit()
            return np.zeros(35)
    else:
        print("No file selected.")

if __name__ == "__main__":
    x = select_and_resize_image()
    x2 = select_and_resize_image()
    x.insert(0,1)
    x2.insert(0,1)
    w = np.random.rand(36)

    xr = np.array([x,x2])

    valores = []
    letra = ''

    yd = np.array([1, 0])

    # Tasa de aprendizaje y número máximo de épocas.
    alfa = 0.004
    max_epochs = 1000

    # Almacenar valores de peso y errores para cada época.
    weight_history = np.zeros((max_epochs, 36))
    error_history = np.zeros((max_epochs, 2))

    for epoch in range(max_epochs):
        error_sum = np.zeros(2)
        for i in range(2):
            tsum = np.dot(w, xr[i, :])
            y0 = (tsum > 0).astype(int)  # Función de activación de umbral
            error = yd[i] - y0
            delta_w = alfa * error * xr[i, :]
            w = w + delta_w
            error_sum += np.abs(error)
    
    # Almacenar valores de peso y errores para esta época.
        weight_history[epoch, :] = w
        error_history[epoch, :] = error_sum
    
    # Verifique la convergencia (el error es cero)
        if np.all(error_sum == 0):
            break

    print('Epochs:', epoch + 1)
    print('Final Weights:')
    print(w)

    imgInput = select_and_resize_image()
    imgInput.insert(0,1)
    comp = np.dot(imgInput, w)
    if comp > 0:
        letra = 'A'
    else:
        letra = 'E'

    print (letra)

    nueva_ventana = Tk()
    nueva_ventana.title("Resultado de la comprobacion")

    etiqueta_letra = tk.Label(nueva_ventana, text=f"La letra ingresada es: {letra}", font=("Arial", 80))
    etiqueta_letra.pack(padx=10, pady=5)
    nueva_ventana.mainloop()





