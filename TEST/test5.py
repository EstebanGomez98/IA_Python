import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

# Función para cargar imágenes y preprocesarlas
def load_and_preprocess_image(file_path):
    img = Image.open(file_path).convert('L')  # Abre la imagen y conviértela a escala de grises
    img = img.resize((28, 28))  # Redimensiona la imagen a 28x28 píxeles
    img = np.array(img)  # Convierte la imagen en una matriz numpy
    img = img.flatten()  # Aplana la matriz en un vector de 784 elementos
    img = img / 255.0  # Normaliza los valores de píxeles (escala de 0 a 1)
    return img

# Función para entrenar una red neuronal y guardar el modelo
def train_neural_network(training_data, labels, model_filename):
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, activation='relu', solver='adam',
                        random_state=1)
    clf.fit(training_data, labels)
    joblib.dump(clf, model_filename)
    return clf

# Función para cargar el modelo entrenado
def load_trained_model(model_filename):
    clf = joblib.load(model_filename)
    return clf

# Función para predecir letras usando la red neuronal
def predict_letter(model, image):
    letter_mapping = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
    predicted_label = model.predict([image])
    predicted_letter = letter_mapping[predicted_label[0]]
    return predicted_letter

# Función para manejar la acción del botón "Cargar Imagen"
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
        prediction_label.config(text=f"Letra predicha: {letter}")

# Función para manejar la acción del botón "Entrenar Red Neuronal"
def train_network():
    training_data = []  # Debe llenarse con datos de entrenamiento
    labels = []  # Debe llenarse con las etiquetas correspondientes
    neural_network = train_neural_network(training_data, labels, "neural_network_model.pkl")
    messagebox.showinfo("Entrenamiento completado", "La red neuronal ha sido entrenada y el modelo se ha guardado.")

# Crear la ventana principal
root = tk.Tk()
root.title("Reconocimiento de Letras")

# Crear la interfaz de usuario
load_button = tk.Button(root, text="Cargar Imagen", command=load_image)
train_button = tk.Button(root, text="Entrenar Red Neuronal", command=train_network)
image_label = tk.Label(root)
prediction_label = tk.Label(root, text="Letra predicha: ")

load_button.pack()
train_button.pack()
image_label.pack()
prediction_label.pack()

# Cargar una red neuronal entrenada (si existe)
neural_network = load_trained_model("neural_network_model.pkl")

root.mainloop()

pip install tkinter numpy matplotlib pillow scikit-learn
