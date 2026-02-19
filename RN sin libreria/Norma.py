import math

def norm(v):
    total = 0 
    for x in v:
        total += x**2
    return math.sqrt(total)

v = [3,4]
w = [2,1]

print("norma de v:", norm(v))
print("norma de w:", norm(w))


