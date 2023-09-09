import numpy as np

# Función para inicializar los pesos en el rango [-1, 1]


def inicializar_pesos(num_entradas):
    return np.random.uniform(-1, 1, num_entradas)

# Función de entrenamiento


def entrenar_adaline(X, YD, alpha, precision):
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

    # Entrenar el perceptrón
    pesos_entrenados = entrenar_adaline(X3, YD3, alpha, precision)

    # Imprimir los pesos entrenados
    print("Pesos entrenados:", pesos_entrenados)

    # Datos para análisis
    datos_analisis = np.array([[1, 1, 0, 1]])

    # Utilizar el perceptrón entrenado para analizar datos
    resultados_analisis = analizar_datos(datos_analisis, pesos_entrenados)

    # Salidas redodeadas
    valores_redondeados = [round(valor, 0) for valor in resultados_analisis]

    # Imprimir los resultados del análisis
    print("Resultados del análisis:", valores_redondeados)
