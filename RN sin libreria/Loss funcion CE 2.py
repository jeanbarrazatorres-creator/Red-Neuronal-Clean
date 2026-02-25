import numpy as np

np.random.seed(0)

# =========================
# Inicialización
# =========================

# Capa 1 (2 entradas → 4 neuronas)
W1 = np.random.randn(4, 2)
B1 = np.random.randn(4)

# Capa 2 (4 → 3 clases)
W2 = np.random.randn(3, 4)
B2 = np.random.randn(3)

# =========================
# Activaciones
# =========================

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    # estabilidad numérica
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

# =========================
# Entrada y valor real
# =========================

x = np.array([1.0, 0.0])  

# Clase correcta = clase 1
y = np.array([0, 1, 0])   # one-hot

# =========================
# Forward pass
# =========================

# Capa 1
z1 = np.dot(W1, x) + B1
a1 = ReLU(z1)

# Capa 2
z2 = np.dot(W2, a1) + B2
y_hat = softmax(z2)

# =========================
# Cross Entropy Loss
# =========================

epsilon = 1e-15
y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

loss = -np.sum(y * np.log(y_hat))

# =========================
# Resultado
# =========================

print("Probabilidades:", y_hat)
print("Clase correcta:", np.argmax(y))
print("Loss:", loss)