def punto(v,w):
    result = 0 
    for i in range(len(v)):
        result += v[i]*w[i]
    return result 
v = [3,4]
w = [2,1]
print("El producto punto:", punto(v,w))