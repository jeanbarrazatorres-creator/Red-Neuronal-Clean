import numpy as np

np.random.seed(0)

# =========================
# Inicialización
# =========================

# Capa 1 (2 entradas → 3 neuronas)
W1 = np.random.randn(3, 2)
B1 = np.random.randn(3)

# Capa 2 (3 → 1 salida)
W2 = np.random.randn(1, 3)
B2 = np.random.randn(1)

# =========================
# Activaciones
# =========================

def ReLU(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# Entrada y valor real
# =========================

x = np.array([1.0, 0.0])   # 2 entradas
y = 1                      # Clasificación binaria (0 o 1)

# =========================
# Forward pass
# =========================

# Capa 1
z1 = np.dot(W1, x) + B1
a1 = ReLU(z1)

# Capa 2
z2 = np.dot(W2, a1) + B2
y_hat = sigmoid(z2)   # Convertimos a probabilidad

# =========================
# Cross Entropy Loss
# =========================

epsilon = 1e-15
y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# =========================
# Resultado
# =========================

print("Probabilidad predicha:", y_hat)
print("Valor real:", y)
print("Cross Entropy Loss:", loss)