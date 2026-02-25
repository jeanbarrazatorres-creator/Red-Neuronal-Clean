import numpy as np 

#CAPA 
W1 = np.random.randn(3,2)
B1 = np.random.randn(3)

W2 = np.random.randn(1,3)
B2 = np.random.randn(1)

#ACTIVACION 
def ReLU(z):
    return np.maximum(0,z)

#ENTRADA Y VALOR REAL 
x = np.array([1,0])
y_real = np.array([1])

#FORWARD pass

z1 = np.dot(W1,x) + B1
a1 = ReLU(z1)

z2 = np.dot(W2,a1) + B2

#LOSS MSE

loss = (z2 - y_real)**2

print("Predicción:", z2)
print("Valor real:", y_real)
print("Loss:", loss)