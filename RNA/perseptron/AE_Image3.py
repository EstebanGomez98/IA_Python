import PySimpleGUI as sg
import numpy as np
from PIL import Image

# Initialize variables for training
w = np.random.rand(36)
xr = []
yd = []

# Define el tamaño de la fuente
font_size = 14  # Puedes ajustar el tamaño de la fuente aquí

# Define the layout for the GUI
layout = [
    [sg.Text("Seleccione las imagenes para el entrenamiento", font=("AnyFont", font_size))],
    [sg.Button("Train Network", font=("AnyFont", font_size)), sg.Button("Select Image", font=("AnyFont", font_size)), sg.Button("Exit", font=("AnyFont", font_size))],
    [sg.Text("", size=(30, 1), key="-OUTPUT-")],
]

# Create the window
window = sg.Window("RNA A y E", layout)

# Training function
def train_network():
    global w

    window["Train Network"].update(disabled=True)

    x = np.array([98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98])
    x2 = np.array([13, 13, 13, 13, 13, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 255, 255, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 13, 13])
    sg.popup("Seleccione la imagen de la letra A")
    x = select_image()
    sg.popup("Seleccione la imagen de la letra E")
    x2 = select_image()
    w = np.random.rand(36)
    x.insert(0, 1)
    x2.insert(0, 1)
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

    sg.popup("Entrenamiento completado")

# Image selection function
def select_image():
    root = sg.tk.Tk()
    root.withdraw()

    file_path = sg.popup_get_file("Select an image file", file_types=(("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"),))

    if file_path:
        try:
            original_image = Image.open(file_path)
            resized_image = original_image.resize((5, 7))
            resized_image = resized_image.convert("L")
            pixel_data = list(resized_image.getdata())
            return pixel_data
        except Exception as e:
            sg.popup_error("Error processing the image.")
            return np.zeros(35)
    else:
        sg.popup("No file selected.")

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    elif event == "Train Network":
        train_network()

    elif event == "Select Image":
        img_data = select_image()
        if img_data:
            img_data.insert(0, 1)
            comp = np.dot(img_data, w)
            if comp > 0:
                letter = 'A'
            else:
                letter = 'E'
            window["-OUTPUT-"].update(f"Predicted Letter: {letter}")

# Close the window
window.close()
