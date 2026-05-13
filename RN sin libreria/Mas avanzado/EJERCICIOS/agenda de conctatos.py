print("Bienvenido a su agenda de contacto")
contactos = {
    "Juan": {
        "telefono": "123456789",
        "correo": "juan@mail.com"
    },
    "Ana": {
        "telefono": "987654321",
        "correo": "ana@mail.com"
    },
    "Pedro": {
        "telefono": "555444333",
        "correo": "pedro@mail.com"
    },
    "Maria": {
        "telefono": "111222333",
        "correo": "maria@mail.com"
    }
}

while True:
    print("1 agregar conctato")
    print("2 buscar conctato")
    print("3 eliminar conctato")
    print("4 mostra todos los contacto")
    print("5 salir")

    try:
        opcion = int(input("Eliga su opcion: "))
    except ValueError:
        print("Esa no es una opcion valida")
        continue

    if opcion == 1:
        while True:

            nombre = input("escribe el nombre del conctatos")
            correo = input("Escribe el correo del contacto")
            numero = input("Escribe el numero del conctato")

            if nombre not in contactos:
                contactos[nombre] = {
                    "correo": correo,
                    "numero": numero
                }
            else:
                print("Ese numero ya esta en tu conctatos")
            
            salir = input("si = salir \ no = no salir").lower()

            if salir == "si":
                print("Saliendo del programa")
                break 
            if salir == "no":
                continue
            else:
                print("Saliendo")

    if opcion == 2:
        while True:
            nombre = input("Nombre del conctato que quieres agregar")

            if nombre in contactos:
                print("Nombre: ", nombre)
                print("Coreo: ", contactos[nombre]["correo"])
                print("Telefono: ", contactos[nombre]["telefono"])
                break
            else:
                print("ese numero no esta")

            salir = input("Si quiere salir diga si ,  si no quiere diga no, si no marca niguna el progrma se cerara: ").lower()

            if salir == "si":
                print("saliendo del programa")
                break
            if salir == "no":
                continue
            else:
                print("saliedo del programa")
                break

    if opcion == 3:
        while True:
            nombre = input("Ecribe el nombre del contacto que quieres eliminar: ").capitalize()

            if nombre in contactos:
                del contactos[nombre]
                print("El conctato", nombre, "fue eliminado")
            else:
                print("Ese conctato no se encontra")

            salir = input(" salir si = a salir / no = no salir / si no es valida la repuesta el programa se cerrara: ").lower()

            if salir == "si":
                print("Salir del programa")
                break
            if salir == "no":
                continue
            else:
                print("Opciones no valida")
                print("Cerrando")
                break

    if opcion == 4:
        print("Tu conctatos son", contactos)

    if opcion == 5:
        print("Cerrando el programa")
        break

    if opcion > 5: 
        print("Esa opcion no es valida")        

            