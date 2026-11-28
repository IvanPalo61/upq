
# motor_chatbot.py - Integrante 4
# Clasificacion de preguntas y motor del chatbot UPQ.

import os

import joblib
import numpy as np

from respuestas import (
    MENSAJE_FALLBACK,
    obtener_respuesta,
    normalizar_texto
)


# ==========================================
# CONFIGURACION DEL MODELO
# ==========================================

BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(
    BASE,
    "modelos",
    "modelo_svm.joblib"
)

UMBRAL_CONFIANZA = 0.25


# ==========================================
# CARGAR MODELO ENTRENADO
# ==========================================

if not os.path.isfile(RUTA_MODELO):

    raise FileNotFoundError(
        "No se encontro modelo_svm.joblib. "
        "Ejecuta primero python modelo_svm.py"
    )

_modelo = joblib.load(RUTA_MODELO)


# ==========================================
# SALUDOS
# ==========================================

SALUDOS = {
    "hola",
    "oye",
    "una duda",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "buenas",
    "que tal",
    "hey",
    "holi",
    "hola buenas",
    "hola buenos dias",
    "hola buenas tardes",
    "hola buenas noches",
    "hola que tal",
    "hola como estas"
}


# ==========================================
# DESPEDIDAS
# ==========================================

DESPEDIDAS = {
    "adios",
    "hasta luego",
    "nos vemos",
    "hasta pronto",
    "bye",
    "hasta mañana",
    "hasta la proxima"
}


# ==========================================
# AGRADECIMIENTOS
# ==========================================

AGRADECIMIENTOS = {
    "gracias",
    "muchas gracias",
    "te lo agradezco",
    "gracias por la ayuda",
    "muchas gracias por la informacion",
    "gracias por responder"
}


# ==========================================
# CLASIFICACION DE INTENCIONES
# ==========================================

def clasificar(pregunta):

    """Devuelve la intencion y el puntaje del modelo."""

    texto = [normalizar_texto(pregunta)]

    if hasattr(_modelo, "decision_function"):

        # Clasificacion mediante SVM
        puntajes = _modelo.decision_function(texto)[0]

    else:

        # Compatibilidad con Naive Bayes
        puntajes = _modelo.predict_proba(texto)[0]

    i = int(np.argmax(puntajes))

    intencion = _modelo.classes_[i]

    confianza = float(puntajes[i])

    return intencion, confianza


# ==========================================
# GENERAR RESPUESTA FALLBACK
# ==========================================

def generar_fallback(confianza=0.0):

    return {
        "intencion": None,
        "confianza": confianza,
        "respuesta": MENSAJE_FALLBACK,
        "fallback": True
    }


# ==========================================
# MOTOR PRINCIPAL DEL CHATBOT
# ==========================================

def responder(pregunta, umbral=UMBRAL_CONFIANZA):

    """Devuelve intencion, confianza, respuesta y fallback."""

    # 1. Validar preguntas vacias
    if not pregunta or not pregunta.strip():

        return {
            "intencion": None,
            "confianza": 0.0,
            "respuesta": (
                "Escribe una pregunta para poder ayudarte."
            ),
            "fallback": True
        }

    # 2. Normalizar la pregunta
    texto_usuario = normalizar_texto(pregunta)

    # 3. Validar que exista al menos una letra
    if not any(c.isalpha() for c in texto_usuario):

        return generar_fallback()

    # 4. Reconocer saludos
    if texto_usuario in SALUDOS:

        return {
            "intencion": "saludo",
            "confianza": 1.0,
            "respuesta": (
                "Hola, soy el asistente virtual de la UPQ, "
                "creado por el grupo TIID 212. "
                "¿En qué puedo ayudarte?"
            ),
            "fallback": False
        }

    # 5. Reconocer despedidas
    if texto_usuario in DESPEDIDAS:

        return {
            "intencion": "despedida",
            "confianza": 1.0,
            "respuesta": (
                "Hasta luego. Gracias por utilizar "
                "el asistente virtual de la UPQ."
            ),
            "fallback": False
        }

    # 6. Reconocer agradecimientos
    if texto_usuario in AGRADECIMIENTOS:

        return {
            "intencion": "agradecimiento",
            "confianza": 1.0,
            "respuesta": (
                "De nada. Si tienes otra pregunta sobre "
                "la UPQ, estoy para ayudarte."
            ),
            "fallback": False
        }

    # 7. Clasificar mediante el modelo SVM
    intencion, confianza = clasificar(texto_usuario)

    # 8. Comprobar el umbral de confianza
    if confianza < umbral:

        return generar_fallback(confianza)

    # 9. Recuperar respuesta mediante similitud coseno
    respuesta = obtener_respuesta(
        intencion,
        texto_usuario
    )

    # 10. Validar que exista una respuesta adecuada
    if not respuesta or respuesta == MENSAJE_FALLBACK:

        return generar_fallback(confianza)

    # 11. Devolver respuesta final
    return {
        "intencion": intencion,
        "confianza": confianza,
        "respuesta": respuesta,
        "fallback": False
    }


# ==========================================
# PRUEBAS DESDE LA TERMINAL
# ==========================================

if __name__ == "__main__":

    print("======================================")
    print("       ASISTENTE VIRTUAL UPQ")
    print("======================================")
    print("Escribe 'salir' para terminar.")

    try:

        while True:

            q = input("\nTú: ").strip()

            if q.lower() == "salir":

                print("Bot: Hasta luego.")
                break

            r = responder(q)

            print(
                f"[{r['intencion']} | confianza "
                f"{r['confianza']:.2f} | "
                f"fallback={r['fallback']}]"
            )

            print("Bot:", r["respuesta"])

    except (KeyboardInterrupt, EOFError):

        print("\nBot: Hasta luego.")
