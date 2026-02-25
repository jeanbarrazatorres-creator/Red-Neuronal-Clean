import numpy as np

np.random.seed(0)

# Inicialización

W1 = np.random.randn(3,2)
B1 = np.random.randn(3)

W2 = np.random.randn(1,3)
B2 = np.random.randn(1)

learning_rate = 0.1

# Activacion

def ReLU(z):
    return np.maximum(0,z)

def ReLU_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Datos

x = np.array([1.0, 0.0])
y = 1

# FORWARD

z1 = np.dot(W1,x) + B1
a1 = ReLU(z1)

z2 = np.dot(W2,a1) + B2
y_hat = sigmoid(z2)

loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

print("Loss antes:", loss)


# BACKWARD


# Capa salida
dz2 = y_hat - y
dW2 = dz2.reshape(1,1) @ a1.reshape(1,3)
dB2 = dz2

# Propagación atrás
da1 = np.dot(W2.T, dz2)

dz1 = da1 * ReLU_derivative(z1)
dW1 = dz1.reshape(3,1) @ x.reshape(1,2)
dB1 = dz1

# UPDATE

W2 -= learning_rate * dW2
B2 -= learning_rate * dB2

W1 -= learning_rate * dW1
B1 -= learning_rate * dB1

# FORWARD NUEVO

z1 = np.dot(W1,x) + B1
a1 = ReLU(z1)

z2 = np.dot(W2,a1) + B2
y_hat = sigmoid(z2)

loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

print("Loss después:", loss)