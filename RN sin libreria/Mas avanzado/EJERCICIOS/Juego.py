import random 
numero = random.randint(1, 100)
print("Bienvenido al adivinador de nuemro")

errores = 0
errores += 1 
while True: 
    print("intento", errores)
    try:
        intento = int(input("Escribe tu nemero:"))
    except ValueError:
        print("ese no es un numero ")
    
    if numero > intento:
        print("El numero es mas alto")
        errores += 1 
        
    elif numero < intento:
        print("El numero es mas bajo")
        errores += 1 

    elif numero == intento:
        print("Ese es el nuemro")
        break 
    if errores == 6:
        print("se te acabaron los intentos")
        print("El numero secreto es", numero)
        break 