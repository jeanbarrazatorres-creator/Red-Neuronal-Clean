print("Bien venido a su cuenta")

saldo = 2000 
print("Su saldo es de :", saldo)

print("Que quire hacer: ")
print("1 depositar")
print("2 retirar")
print("3 salir")

Elegir = input("Eliga su decion: ")

if Elegir == "1":
    Nombre_de_la_cuenta = input("Escriba el nombre de la cuenta: ")
    Monto = int(input("Escriba el monto: "))
    saldo_actu = saldo - Monto
    print("Se depositaron:", Monto, "a esta cuenta:", Nombre_de_la_cuenta)
    print("Tu saldo actual es:", saldo_actu)
elif Elegir == "2": 
    Retirar = int(input("Cuanto quieres retirar: "))
    saldo_actu2 = saldo - Retirar
    print("Retiraste:", Retirar)
    print("saldo actual es:", saldo_actu2)
elif Elegir == "3":
    print("Adios")
else: 
    print("Ese opcion no es valida")