import numpy as np
import matplotlib.pyplot as plt

# =========================
# Datos AND
# =========================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# =========================
# Inicialización
# =========================

w = np.random.randn(2)
b = 0
lr = 5

def step(z):
    return 1 if z >= 0 else 0

# =========================
# Entrenamiento
# =========================

for epoch in range(20):
    for i in range(len(X)):
        z = np.dot(w, X[i]) + b
        y_pred = step(z)

        error = y[i] - y_pred

        w += lr * error * X[i]
        b += lr * error

print("Pesos finales:", w)
print("Bias final:", b)

# =========================
# Graficar puntos
# =========================

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='o')
    else:
        plt.scatter(X[i][0], X[i][1], marker='x')

# =========================
# Dibujar frontera
# =========================

x_values = np.linspace(-0.5, 1.5, 100)

# y = -(w1/w2)x - b/w2
y_values = -(w[0]/w[1]) * x_values - (b/w[1])

plt.plot(x_values, y_values)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.title("Frontera de decisión - Perceptrón AND")
plt.show()