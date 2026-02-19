#Para sacar los angulos se nesecitan las tres

import math 

def punto(v,w):
    result = 0 
    for i in range(len(v)):
        result += v[i]*w[i]
    return result 
v = [3,4]
w = [2,1]
print("El producto punto:", punto(v,w))

def norm(v):
    total = 0 
    for x in v:
        total += x**2
    return math.sqrt(total)

print("norma de v:", norm(v))
print("norma de w:", norm(w))

# Esta es la funcion de sacar angulos 

def angle(v,w):
    dot = punto(v,w)
    return math.acos(dot / (norm(v) * norm(w)))

print("Angulo (radiantes):", angle(v,w))