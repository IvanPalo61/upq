##print("Mi proyecto de Machine Learning UPQ funciona")
#Comentario 
# Cambio 1

from motor_chatbot import responder

preguntas = [
    "¿Dónde está la UPQ?",
    "¿Qué carreras ofrece la UPQ?",
    "¿Hay becas?",
    "¿Quién ganó el mundial?",
    "¿Cuánto mide la Muralla China?",
    "¿Cómo cocinar mole?",
    "¿Te puedo preguntar algo?",
    "Oye"
]

for p in preguntas:
    r = responder(p)

    print("\n" + "="*50)
    print("Pregunta:", p)
    print("Respuesta:", r["respuesta"])
    print("Intención:", r["intencion"])
    print("Confianza:", r["confianza"])
    print("Fallback:", r["fallback"])
