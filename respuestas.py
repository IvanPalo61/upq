# respuestas.py - Integrante 4
# Relaciona cada intención con su respuesta oficial del dataset.
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modelo_svm import leer_dataset

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(BASE, "dataset_upq.csv")

MENSAJE_FALLBACK = (
    "Lo siento, no estoy seguro de haber entendido tu pregunta. "
    "Intenta reformularla o consulta www.upq.mx / (442) 101 9000."
)

_df = leer_dataset(RUTA_DATASET)
_df["texto_norm"] = _df["texto"].str.lower().str.strip()

# TF-IDF solo para elegir, dentro de la intención predicha,
# la respuesta de la pregunta más parecida (varias intenciones tienen
# respuestas distintas según la pregunta).
_vec = TfidfVectorizer(strip_accents="unicode").fit(_df["texto_norm"])
_mat = _vec.transform(_df["texto_norm"])


def intenciones():
    return sorted(_df["intencion"].unique())


def respuesta_por_intencion(intencion):
    """Respuesta 'general' de la intención (la primera registrada)."""
    filas = _df[_df["intencion"] == intencion]
    return filas["respuesta"].iloc[0] if len(filas) else MENSAJE_FALLBACK


def obtener_respuesta(intencion, pregunta):
    """Respuesta oficial más adecuada dentro de la intención predicha."""
    idx = _df.index[_df["intencion"] == intencion].tolist()
    if not idx:
        return MENSAJE_FALLBACK
    sim = cosine_similarity(_vec.transform([pregunta.lower().strip()]), _mat[idx])[0]
    return _df.loc[idx[sim.argmax()], "respuesta"]