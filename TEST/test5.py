import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import os

training_folder = "input_images"
# Function to load and preprocess images
def load_and_preprocess_image(file_path):
    img = Image.open(file_path).convert('L')  
    img = img.resize((28, 28))  
    img = np.array(img)  
    img = img.flatten()  
    img = img / 255.0  
    return img

# Function to train a neural network and save the model
def train_neural_network(training_data, labels, model_filename):
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, activation='relu', solver='adam', random_state=1)
    clf.fit(training_data, labels)
    joblib.dump(clf, model_filename)
    return clf

# Function to load a trained model
def load_trained_model(model_filename):
    clf = joblib.load(model_filename)
    return clf

# Function to predict letters using the neural network
def predict_letter(model, image):
    letter_mapping = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
    predicted_label = model.predict([image])[0]
    
    # Add debugging print statement
    print(f"Predicted Label: {predicted_label}")
    
    if predicted_label in letter_mapping:
        predicted_letter = letter_mapping[predicted_label]
    else:
        predicted_letter = predicted_label
    
    return predicted_letter


# Function to handle the "Load Image" button action
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = load_and_preprocess_image(file_path)
        img = Image.open(file_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        letter = predict_letter(neural_network, image)
        prediction_label.config(text=f"Predicted Letter: {letter}")

def load_training_data():
    training_data = []
    labels = []

    # Define the list of letters
    letters = ['A', 'E', 'I', 'O', 'U']

    # Load 5 images for each letter
    for letter in letters:
        for i in range(1, 6):  # Load A1, A2, A3, A4, A5 for 'A', and so on
            image_path = os.path.join(training_folder, f"{letter}{i}.jpg")
            if os.path.exists(image_path):
                image = load_and_preprocess_image(image_path)
                training_data.append(image)
                labels.append(letter)

    return training_data, labels

# Function to handle the "Train Neural Network" button action
def train_network():
    global neural_network
    training_data, labels = load_training_data()
    
    if len(training_data) == 0 or len(labels) == 0:
        messagebox.showerror("Error", "Training data and labels are empty.")
        return

    neural_network = train_neural_network(training_data, labels, "neural_network_model.pkl")
    messagebox.showinfo("Training Completed", "The neural network has been trained and the model has been saved.")

# Create the main window
root = tk.Tk()
root.title("Letter Recognition")

# Create the user interface elements
load_button = tk.Button(root, text="Load Image", command=load_image)
train_button = tk.Button(root, text="Train Neural Network", command=train_network)
image_label = tk.Label(root)
prediction_label = tk.Label(root, text="Predicted Letter: ")

load_button.pack()
train_button.pack()
image_label.pack()
prediction_label.pack()

# Load a trained neural network (if it exists)
neural_network = None

root.mainloop()
