import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt  # Importamos matplotlib

font_size = 14  # Puedes ajustar el tamaño de la fuente aquí

# Define the layout for the GUI
layout = [
    [sg.Text("Click 'Entrenar' para entrenar el modelo", font=("AnyFont", font_size))],
    [sg.Button("Entrenar", font=("AnyFont", font_size))],
    [sg.Text("Click 'Reconocer' para reconocer los bits", font=("AnyFont", font_size))],
    [sg.Button("Reconocer", font=("AnyFont", font_size))],
    [sg.Button("Finalizar", font=("AnyFont", font_size))]
]

# Función para inicializar los pesos en el rango [-1, 1]


def inicializar_pesos(num_entradas):
    return np.random.uniform(-1, 1, num_entradas)

# Función de entrenamiento


def entrenar_perceptron(X, YD, alpha, precision):
    num_entradas = X.shape[1]
    num_muestras = X.shape[0]

    # Inicializar pesos
    W = inicializar_pesos(num_entradas)

    # Inicializar cuenta de épocas
    epocas = 0

    # Inicializar error anterior
    error_anterior = float('inf')

    while True:
        # Inicializar el error para esta época
        error_actual = 0

        for i in range(num_muestras):
            # Calcular la suma ponderada de los patrones
            Yo = np.dot(W, X[i])

            # Actualizar pesos
            W = W + alpha * (YD[i] - Yo) * X[i]

            # Calcular el error para este patrón
            error_actual += (YD[i] - Yo) ** 2

        # Calcular |E(t) - E(t-1)|
        error_diff = abs(error_actual - error_anterior)

        # Actualizar el error anterior
        error_anterior = error_actual

        # Incrementar el contador de épocas
        epocas += 1

        # Calcular E(actual)
        error_actual /= num_muestras

        # Imprimir información de la época actual
        print(f"Época {epocas}: Error = {error_actual}")         
        
        # Verificar si se ha alcanzado la precisión deseada
        if error_diff < precision:
            print("Entrenamiento completado.")                  
            break
            
    return W

# Función para analizar datos utilizando el perceptrón entrenado


def analizar_datos(datos, pesos_entrenados):
    num_muestras = datos.shape[0]
    resultados = []

    for i in range(num_muestras):
        # Calcular la suma ponderada de los patrones
        resultado = np.dot(pesos_entrenados, datos[i])
        resultados.append(resultado)
    print(resultados)
    return resultados


# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrada (X) y salida deseada (YD)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1],])

    YD = np.array([0, 1, 2, 3])

    X2 = np.array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]])

    YD2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    X3 = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 1, 1],
                   [0, 1, 0, 0],
                   [0, 1, 0, 1],
                   [0, 1, 1, 0],
                   [0, 1, 1, 1],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 1, 0],
                   [1, 0, 1, 1],
                   [1, 1, 0, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]])

    YD3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # Parámetros de entrenamiento
    alpha = 0.1
    precision = 0.001
    
    window = sg.Window("Adelaine Convertidor Binario a Decimal", layout)

    # Event loop
    while True:
        event, values = window.read()
        
        if event == sg.WINDOW_CLOSED or event == "Finalizar":
            break
        
        if event == "Entrenar":
            train_options = ["2 bit", "3 bit", "4 bit", "No entrenar"]
            #choice = sg.popup("Seleccione la opcion de entrenamiento", custom_text=(train_options[0], train_options[1], train_options[2], train_options[3]))
            layout_options = [
                [sg.Text("Seleccione una opcion:")],
                [sg.Radio("2 bit", "Bits", default=True, key="Option1")],
                [sg.Radio("3 bit", "Bits", key="Option2")],
                [sg.Radio("4 bit", "Bits", key="Option3")],
                [sg.Button("Seleccionar"), sg.Button("Cerrar")]
                ]

# Create the window
            window_options = sg.Window("Select an Option", layout_options)

# Event loop
            while True:
                event, values = window_options.read()
                if event == sg.WINDOW_CLOSED or event == "Cerrar":
                    break
                if event == "Seleccionar":
                    # Check which option was selected
                    selected_option = None
                    for i in range(1, 5):
                        if values[f"Option{i}"]:
                            selected_option = f"Option{i}"
                            break                   
                    sg.popup(f"Selecciono {selected_option}")
            # Close the window
            window_options.close()
        
            if selected_option:
                if selected_option == "Option1":
                    pesos_entrenados = entrenar_perceptron(X, YD, alpha, precision)
                    global bits
                    bits = 2 
                elif selected_option == "Option2":
                    pesos_entrenados = entrenar_perceptron(X2, YD2, alpha, precision)
                    bits = 3
                elif selected_option == "Option3":
                    pesos_entrenados = entrenar_perceptron(X3, YD3, alpha, precision)
                    bits = 4
            sg.popup(f"Entrenamiento con '{selected_option}' completado.")
        
        elif event == "Reconocer":
            
            input_layout2 = [
                [sg.Text("Digite los bits:")],
                [sg.InputText(), sg.InputText()],
                [sg.Button("OK")]
            ]
            input_layout3 = [
                [sg.Text("Digite los bits:")],
                [sg.InputText(), sg.InputText(), sg.InputText()],
                [sg.Button("OK")]
            ]
            input_layout4 = [
                [sg.Text("Digite los bits:")],
                [sg.InputText(), sg.InputText(), sg.InputText(), sg.InputText()],
                [sg.Button("OK")]
            ]
            if bits == 2:
                input_window = sg.Window("Digite bit por bit", input_layout2)
            elif bits == 3:
                input_window = sg.Window("Digite bit por bit", input_layout3)
            else:
                input_window = sg.Window("Digite bit por bit", input_layout4)
            
            while True:
                input_event, input_values = input_window.read()
                if input_event == sg.WINDOW_CLOSED or input_event == "OK":
                    break
            input_window.close()
            
            if input_event == "OK":
                
                # Convert the entered numbers to an array
                if bits == 2:
                    num_array = np.array([[float(input_values[0]), float(input_values[1])]])
                elif bits == 3:
                    num_array = np.array([[float(input_values[0]), float(input_values[1]), float(input_values[2])]])
                else:
                    num_array = np.array([[float(input_values[0]), float(input_values[1]), float(input_values[2]) , float(input_values[3])]])
                
                # Perform recognition using the entered numbers (replace with your code)
                resultados_analisis = analizar_datos(num_array, pesos_entrenados)
                valores_redondeados = [round(valor, 0) for valor in resultados_analisis]
                sg.popup(f"Reconocimiento completado, el resultado es: ", valores_redondeados)

    # Close the window
    window.close()
