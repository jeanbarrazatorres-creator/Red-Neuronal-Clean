import numpy as np 

#Capa de neurona 
W = np.random.randn(3,2)
b = np.random.randn(3)

x = np.array([1,0]) #entrada 
z = np.dot(W,x) + b 
print("salida de la capa:", z)