# motor_chatbot.py - Integrante 4

# Carga el modelo, clasifica la pregunta, aplica umbral de confianza
# (fallback) y devuelve la respuesta oficial.

import os

import joblib
import numpy as np

from respuestas import MENSAJE_FALLBACK, obtener_respuesta

BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(
    BASE,
    "modelos",
    "modelo_svm.joblib"
)

# Confianza = puntaje máximo de decision_function del SVM.
# En pruebas: preguntas de la UPQ ≈ 0.6-1.3;
# fuera de tema ≈ -0.7-0.1.

UMBRAL_CONFIANZA = 0.25

_modelo = joblib.load(RUTA_MODELO)


def clasificar(pregunta):
    """Devuelve (intencion, confianza) para una pregunta."""
 
    texto = [pregunta.lower().strip()]

    if hasattr(_modelo, "decision_function"):  # SVM
        puntajes = _modelo.decision_function(texto)[0]
    else:  # Naive Bayes
        puntajes = _modelo.predict_proba(texto)[0]

    i = int(np.argmax(puntajes))

    return _modelo.classes_[i], float(puntajes[i])


def responder(pregunta, umbral=UMBRAL_CONFIANZA):
    """Devuelve dict: intencion, confianza, respuesta, fallback."""

    if not pregunta or not pregunta.strip():

        return {
            "intencion": None,
            "confianza": 0.0,
            "respuesta": "Escribe una pregunta para poder ayudarte.",
            "fallback": True
        }

    texto_usuario = pregunta.lower().strip()

    saludos = [
        "hola"
        "oye",
        "una duda"
    ]
   
    if texto_usuario in saludos:

        return {
            "intencion": "saludo",
            "confianza": 1.0,
            "respuesta": "Hola, soy el asistente virtual de la UPQ creado por el TIID 212. Dime en qué te ayudo?",
            "fallback": False
        }
    
    
    intencion, confianza = clasificar(pregunta)

    if confianza < umbral:

        return {
            "intencion": None,
            "confianza": confianza,
            "respuesta": MENSAJE_FALLBACK,
            "fallback": True
        }

    return {
        "intencion": intencion,
        "confianza": confianza,
        "respuesta": obtener_respuesta(intencion, pregunta),
        "fallback": False
    }


if __name__ == "__main__":

    print("Motor UPQ (escribe 'salir' para terminar)")

    while True:

        q = input("\nTú: ").strip()

        if q.lower() == "salir":
            break

        r = responder(q)

        print(
            f"[{r['intencion']} | confianza "
            f"{r['confianza']:.2f} | "
            f"fallback={r['fallback']}]"
        )

        print("Bot:", r["respuesta"])