import PySimpleGUI as sg
import numpy as np
from PIL import Image
import random

# Initialize variables for training
w = np.random.rand(36)
xr = []
yd = []

# Define the layout for the GUI
layout = [
    [sg.Text("Select an image and train the network")],
    [sg.Button("Train Network"), sg.Button("Select Image"), sg.Button("Exit")],
    [sg.Text("", size=(30, 1), key="-OUTPUT-")],
]

# Create the window
window = sg.Window("Simple GUI for Training", layout)

# Training function
def train_network():
    global w, xr, yd
    # Replace this with your training code
    # For now, let's just generate random data
    w = np.random.rand(36)
    xr = [np.random.rand(36) for _ in range(10)]
    yd = [random.choice([0, 1]) for _ in range(10)]
    sg.popup("Training completed!")

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
