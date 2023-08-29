import numpy as np
import matplotlib.pyplot as plt

# Initializations
w = np.random.rand(36)  # Initialize weights with random values between 0 and 1
x = np.array([
    [1, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 98, 98, 98, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98, 98, 255, 255, 255, 98],
    [1, 13, 13, 13, 13, 13, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 255, 255, 13, 255, 255, 255, 255, 13, 255, 255, 255, 255, 13, 13, 13, 13, 13]
])

yd = np.array([0, 1])

# Learning rate and maximum number of epochs
alfa = 0.004
max_epochs = 1000

# Store weight values and errors for each epoch
weight_history = np.zeros((max_epochs, 36))
error_history = np.zeros((max_epochs, 2))

# Training loop
for epoch in range(max_epochs):
    error_sum = np.zeros(2)
    for i in range(2):
        tsum = np.dot(w, x[i, :])
        y0 = (tsum > 0).astype(int)  # Threshold activation function
        error = yd[i] - y0
        delta_w = alfa * error * x[i, :]
        w = w + delta_w
        error_sum += np.abs(error)
    
    # Store weight values and errors for this epoch
    weight_history[epoch, :] = w
    error_history[epoch, :] = error_sum
    
    # Check for convergence (error is zero)
    if np.all(error_sum == 0):
        break

print('Epochs:', epoch + 1)
print('Final Weights:')
print(w)

# Plot the weight values versus epoch
plt.figure()
for i in range(36):
    plt.plot(range(epoch + 1), weight_history[:epoch + 1, i])
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('Weight Values vs. Epoch')
plt.grid(True)

# Plot the error versus epoch
plt.figure()
for i in range(2):
    plt.plot(range(epoch + 1), error_history[:epoch + 1, i], '-o', label=f'Output {i}')
plt.xlabel('Epoch')
plt.ylabel('Error Sum')
plt.title('Error Sum vs. Epoch')
plt.legend()
plt.grid(True)

plt.show()
