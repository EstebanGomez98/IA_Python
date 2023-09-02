from tkinter import Tk, filedialog, Button
import tkinter as tk
import numpy as np
from PIL import Image

# ... (rest of your code)

# Function to train the network
def train_network():
    # Add your training code here
    # For example:
    # Training code can go here, similar to the training code in your __main__ block
    pass

# Function to scan an image and make a prediction
def scan_image():
    imgInput = select_and_resize_image()
    imgInput.insert(0, 1)
    comp = np.dot(imgInput, w)
    if comp > 0:
        letra = 'A'
    else:
        letra = 'E'

    print(letra)

    # Create a new tkinter window to display the result
    result_window = Tk()
    result_window.title("Result of the Scan")

    label_result = tk.Label(result_window, text=f"The scanned letter is: {letra}", font=("Arial", 80))
    label_result.pack(padx=10, pady=5)
    result_window.mainloop()

# Function to close the system
def close_system():
    root.quit()

# Create the main tkinter window
root = Tk()
root.withdraw()  # Hide the main window

# Create buttons for training, scanning, and closing
train_button = Button(root, text="Train Network", command=train_network)
scan_button = Button(root, text="Scan Image", command=scan_image)
close_button = Button(root, text="Close System", command=close_system)

# Pack the buttons in the main window
train_button.pack()
scan_button.pack()
close_button.pack()

# Run the tkinter main loop
root.mainloop()
