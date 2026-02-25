import numpy as np

#entrada del problema ADN

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

#vector de peso 
w = np.random.rand(2)

#bias
b = 0

#Learning rete
lr = 0.1

#fucion de activacion 

def step(z):
    return 1 if z >= 0 else 0

#entrenamiento 
for epoch in range(20):
    for i in range(len(x)):

        #neurona 

        z = np.dot(w, x[i]) + b 
        y_pred = step(z)

        #calculo del error

        error = y[i] - y_pred

        #actualizacion de pesos y bias 
        w += lr * error * x[i]
        b += lr * error 


#Resultado 
print("Pesos finales:", w)
print("bias finales:", b)

#Probar modelo
print("\nPredicciones finales:")
for i in range(len(x)):
    z = np.dot(w, x[i]) + b 
    y_pred = step(z)
    print(f"Entrada: {x[i]} prediccion:{y_pred}")

