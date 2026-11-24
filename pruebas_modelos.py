from modelo_svm import predecir, leer_dataset
respuestas = leer_dataset().drop_duplicates("intencion").set_index("intencion")["respuesta"]
print("Escribe una pregunta (o 'salir' para terminar)")
while True:
    pregunta = input("\nTú: ").strip()
    if pregunta.lower() == "salir":
        break
    intencion = predecir(pregunta)
    print(f"Intención detectada: {intencion}")
    print(f"Respuesta: {respuestas[intencion]}")
