print("Bienvenido a su cuenta")

saldo = 2000 

while True:
    print("\nSu saldo es:", saldo)

    print("\n¿Qué quiere hacer?")
    print("1. Depositar")
    print("2. Retirar")
    print("3. Salir")

    elegir = input("Elija su opción: ")

    if elegir == "1":
        nombre_cuenta = input("Escriba el nombre de la cuenta: ")
        monto = int(input("Escriba el monto a depositar: "))

        saldo = saldo + monto  

        print("Se depositaron:", monto, "a la cuenta:", nombre_cuenta)
        print("Tu saldo actual es:", saldo)

    elif elegir == "2":
        retirar = int(input("¿Cuánto quieres retirar?: "))

        if retirar > saldo:
            print(" No tienes suficiente saldo")
        else:
            saldo = saldo - retirar
            print("Retiraste:", retirar)
            print("Saldo actual:", saldo)

    elif elegir == "3":
        print("Adiós 👋")
        break

    else:
        print(" Esa opción no es válida")