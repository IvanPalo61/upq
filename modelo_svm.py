#primer modelo de IA SVM
import csv
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RUTA_DATOS = "dataset_upq.csv"
COL_TEXTO = "texto"
COL_ETIQUETA = "intencion"
TEST_SIZE = 0.2
RANDOM_STATE = 42
RUTA_MODELO = "modelos/modelo_svm.joblib"

def leer_dataset(ruta=RUTA_DATOS):
    filas = []
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        f.readline()  
        for linea in f:
            linea = linea.rstrip("\r\n")
            if not linea.strip():
                continue
            if "\t" in linea:
                campos = linea.split("\t")
                while campos and campos[-1] == "":
                    campos.pop()
            else:
                campos = next(csv.reader([linea]))
            if len(campos) == 3:
                filas.append([c.strip() for c in campos])
    return pd.DataFrame(filas, columns=[COL_TEXTO, COL_ETIQUETA, "respuesta"])

def cargar_datos_y_dividir():
    df = leer_dataset().dropna(subset=[COL_TEXTO, COL_ETIQUETA])
    X = df[COL_TEXTO].astype(str).str.lower().str.strip()
    y = df[COL_ETIQUETA]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def entrenar_svm(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode")),
        ("svm", LinearSVC(random_state=RANDOM_STATE)),
    ])
    parametros = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True, False],
        "svm__C": [0.1, 0.5, 1, 2, 5, 10],
        "svm__class_weight": [None, "balanced"],
    }
    busqueda = GridSearchCV(
        pipeline, parametros, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    busqueda.fit(X_train, y_train)
    print("\nMejores parámetros SVM:", busqueda.best_params_)
    print(f"Mejor F1 macro (validación cruzada): {busqueda.best_score_:.4f}")
    return busqueda.best_estimator_


def evaluar(modelo, X_test, y_test, nombre="SVM"):
    pred = modelo.predict(X_test)
    print(f"\n===== Resultados {nombre} =====")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    return pred


def guardar_modelo(modelo, ruta=RUTA_MODELO):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    joblib.dump(modelo, ruta)
    print(f"\nModelo guardado en: {ruta}")


def predecir(pregunta, ruta=RUTA_MODELO):
    modelo = joblib.load(ruta)
    return modelo.predict([pregunta.lower().strip()])[0]


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_datos_y_dividir()
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    modelo = entrenar_svm(X_train, y_train)
    evaluar(modelo, X_test, y_test, "SVM")
    guardar_modelo(modelo)



      





      
