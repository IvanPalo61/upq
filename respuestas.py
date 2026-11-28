
# respuestas.py - Integrante 4
# Recuperacion de respuestas oficiales del dataset.

import os
import re
import unicodedata

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modelo_svm import leer_dataset


# ==========================================
# CONFIGURACION
# ==========================================

BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_DATASET = os.path.join(BASE, "dataset_upq.csv")

UMBRAL_SIMILITUD = 0.25

PESO_PALABRAS = 0.60
PESO_CARACTERES = 0.40

MENSAJE_FALLBACK = (
    "Lo siento, no estoy seguro de haber entendido tu pregunta. "
    "Intenta reformularla o consulta www.upq.mx / (442) 101 9000."
)


# ==========================================
# NORMALIZACION DEL TEXTO
# ==========================================

def normalizar_texto(texto):

    texto = str(texto).lower().strip()

    # Eliminar acentos conservando la letra ñ
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("ñ", "\ue000")

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = texto.replace("\ue000", "ñ")

    # Eliminar signos de puntuacion
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Corregir variantes y errores ortograficos
    equivalencias = {
        "reinscribirse": "reinscripcion",
        "reinscribirme": "reinscripcion",
        "reinscribirte": "reinscripcion",
        "reinscripciones": "reinscripcion",
        "reinscrbirse": "reinscripcion",
        "reinisceribirme": "reinscripcion",
        "reinscrbirme": "reinscripcion",
        "reinscribo": "reinscripcion",
        "reinscribir": "reinscripcion",
        "reinscripsion": "reinscripcion",
        "reincripcion": "reinscripcion",
        "inscripsion": "inscripcion",
        "incripcion": "inscripcion",
        "admisionn": "admision",
        "admicion": "admision",
        "uvicacion": "ubicacion",
        "carrerras": "carreras",
        "vekas": "becas",
        "becaz": "becas"
    }

    palabras = texto.split()

    palabras = [
        equivalencias.get(palabra, palabra)
        for palabra in palabras
    ]

    return " ".join(palabras)


# ==========================================
# CARGAR DATASET
# ==========================================

_df = leer_dataset(RUTA_DATASET)

_df = _df.dropna(
    subset=["texto", "intencion", "respuesta"]
).reset_index(drop=True)

_df["texto_norm"] = _df["texto"].apply(
    normalizar_texto
)


# ==========================================
# TF-IDF POR PALABRAS
# ==========================================

_vec_palabras = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True
)

_mat_palabras = _vec_palabras.fit_transform(
    _df["texto_norm"]
)


# ==========================================
# TF-IDF POR CARACTERES
# ==========================================

_vec_caracteres = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    sublinear_tf=True
)

_mat_caracteres = _vec_caracteres.fit_transform(
    _df["texto_norm"]
)


# ==========================================
# OBTENER INTENCIONES
# ==========================================

def intenciones():

    return sorted(
        _df["intencion"].unique()
    )


# ==========================================
# RESPUESTA GENERAL DE UNA INTENCION
# ==========================================

def respuesta_por_intencion(intencion):

    filas = _df[
        _df["intencion"] == intencion
    ]

    if filas.empty:
        return MENSAJE_FALLBACK

    return filas["respuesta"].iloc[0]


# ==========================================
# RECUPERACION DE RESPUESTAS
# ==========================================

def obtener_respuesta(
    intencion,
    pregunta,
    umbral=UMBRAL_SIMILITUD
):

    texto = normalizar_texto(pregunta)

    if not texto:
        return MENSAJE_FALLBACK

    # Obtener preguntas de la intencion predicha
    idx = _df.index[
        _df["intencion"] == intencion
    ].tolist()

    if not idx:
        return MENSAJE_FALLBACK

    filas = _df.loc[idx]

    # 1. Buscar coincidencias exactas
    exactas = filas[
        filas["texto_norm"] == texto
    ]

    if not exactas.empty:

        respuestas = exactas["respuesta"].unique()

        # Evitar respuestas contradictorias
        if len(respuestas) == 1:
            return respuestas[0]

        return MENSAJE_FALLBACK

    # 2. Vectorizacion por palabras
    pregunta_palabras = _vec_palabras.transform(
        [texto]
    )

    sim_palabras = cosine_similarity(
        pregunta_palabras,
        _mat_palabras[idx]
    )[0]

    # 3. Vectorizacion por caracteres
    pregunta_caracteres = _vec_caracteres.transform(
        [texto]
    )

    sim_caracteres = cosine_similarity(
        pregunta_caracteres,
        _mat_caracteres[idx]
    )[0]

    # 4. Combinar similitudes
    similitud = (
        PESO_PALABRAS * sim_palabras
        + PESO_CARACTERES * sim_caracteres
    )

    # 5. Obtener mejor coincidencia
    mejor_posicion = int(np.argmax(similitud))

    mejor_similitud = float(
        similitud[mejor_posicion]
    )

    # 6. Validar umbral de similitud
    if mejor_similitud < umbral:

        return MENSAJE_FALLBACK

    # 7. Obtener indice de la mejor pregunta
    mejor_indice = idx[mejor_posicion]

    # 8. Recuperar respuesta oficial
    return _df.loc[
        mejor_indice,
        "respuesta"
    ]
