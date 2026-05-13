# Sistema de Notas de Estudiantes

estudiantes = []

while True:
    nombre = input("Ingresa el nombre del estudiante: ")

    # Pedir 3 notas
    nota1 = float(input("Ingresa la nota 1: "))
    nota2 = float(input("Ingresa la nota 2: "))
    nota3 = float(input("Ingresa la nota 3: "))

    # Calcular promedio
    promedio = (nota1 + nota2 + nota3) / 3

    # Determinar estado
    if promedio >= 4.0:
        estado = "Aprobado"
    else:
        estado = "Reprobado"

    # Guardar en lista
    estudiante = {
        "nombre": nombre,
        "promedio": promedio,
        "estado": estado
    }

    estudiantes.append(estudiante)

    print("\n--- Resultado ---")
    print("Nombre:", nombre)
    print("Promedio:", round(promedio, 2))
    print("Estado:", estado)

    # Preguntar si quiere continuar
    continuar = input("\n¿Quieres agregar otro estudiante? (s/n): ")
    if continuar.lower() != "s":
        break

# Buscar el mejor promedio
mejor = estudiantes[0]

for est in estudiantes:
    if est["promedio"] > mejor["promedio"]:
        mejor = est

print("\n=== Mejor Estudiante ===")
print("Nombre:", mejor["nombre"])
print("Promedio:", round(mejor["promedio"], 2))
print("Estado:", mejor["estado"])
