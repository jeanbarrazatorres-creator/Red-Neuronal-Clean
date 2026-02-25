import numpy as np
# Capa 1 

W1 = np.random.randn(3,2)
b1 = np.random.randn(3)

#capa 2 
W2 = np.random.randn(1,3)
b2 = np.random.randn(1)

#funcion de activacion (ReLU)

def ReLU(z):
    return np.maximum(0,z)

#forwaord pass

x = np.array([1,0]) #entrada 

#primer capa 

z1 = np.dot(W1,x) + b1
a1 = ReLU(z1)

#segunda capa 

z2 = np.dot(W2,a1) + b2

#resultado 
print("salida finsal:",z2)