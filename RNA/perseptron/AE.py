import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

# Inicializaciones
w = np.random.rand(36)  # Inicializar pesos con valores aleatorios entre 0 y 1
x = np.array([
    [1, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98],
    [1, 13, 13, 13, 13, 13, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 255, 255, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 13, 13]
])
valores = []
letra = ''

yd = np.array([1, 0])

# Tasa de aprendizaje y número máximo de épocas.
alfa = 0.004
max_epochs = 1000

# Almacenar valores de peso y errores para cada época.
weight_history = np.zeros((max_epochs, 36))
error_history = np.zeros((max_epochs, 2))

# Bucle de entrenamiento
for epoch in range(max_epochs):
    error_sum = np.zeros(2)
    for i in range(2):
        tsum = np.dot(w, x[i, :])
        y0 = (tsum > 0).astype(int)  # Función de activación de umbral
        error = yd[i] - y0
        delta_w = alfa * error * x[i, :]
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

#ventana para 
def obtener_valores():
    for i in range(35):
        valor = float(cajas_texto[i].get())
        valores.append(valor)
    print("Valores ingresados por el usuario:", valores)
    
    # Cerrar la ventana actual
    ventana.destroy()
    
# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Ingresar Valores")

# Crear 35 cajas de texto y etiquetas
cajas_texto = []

for i in range(7):
    for j in range(5):
        caja_texto = tk.Entry(ventana)
        caja_texto.grid(row=i, column=j, padx=2, pady=2)
        

        cajas_texto.append(caja_texto)

# Botón para obtener los valores ingresados
boton_obtener = tk.Button(ventana, text="Obtener Valores", command=obtener_valores)
boton_obtener.grid(row=7, column=0, columnspan=2, pady=10)

# Iniciar el bucle de la ventana
ventana.mainloop()

valores.insert(0, 1)
comp = np.dot(valores, w)
if comp > 0:
    letra = 'A'
else:
    letra = 'E'

print (letra)
    
# Crear una nueva ventana para mostrar el valor de 'letra'
nueva_ventana = tk.Tk()
nueva_ventana.title("Resultado de la comprobacion")

etiqueta_letra = tk.Label(nueva_ventana, text=f"La letra ingresada es: {letra}", font=("Arial", 80))
etiqueta_letra.pack(padx=10, pady=5)

nueva_ventana.mainloop()

# Después de que la ventana tkinter se cierre, mostrar las gráficas
# Trazar los valores de peso versus la época.
plt.figure()
for i in range(36):
    plt.plot(range(epoch + 1), weight_history[:epoch + 1, i])
plt.xlabel('época')
plt.ylabel('valores de peso')
plt.title('valores de peso vs. época')
plt.grid(True)

# Trazar el error versus la época
plt.figure()
for i in range(2):
    plt.plot(range(epoch + 1), error_history[:epoch + 1, i], '-o', label=f'Output {i}')
plt.xlabel('época')
plt.ylabel('Error')
plt.title('Error vs. época')
plt.legend()
plt.grid(True)
plt.show()