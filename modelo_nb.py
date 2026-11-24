import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from modelo_svm import RUTA_MODELO as RUTA_SVM
from modelo_svm import cargar_datos_y_dividir, evaluar

RUTA_MODELO_NB = "modelos/modelo_nb.joblib"


def entrenar_nb(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode")),
        ("nb", MultinomialNB()),
    ])
    parametros = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [True, False],
        "nb__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],  
        "nb__fit_prior": [True, False],
    }
    busqueda = GridSearchCV(
        pipeline, parametros, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1 )
    busqueda.fit(X_train, y_train)
    print("\nMejores parámetros Naive Bayes:", busqueda.best_params_)
    print(f"Mejor F1 macro (validación cruzada): {busqueda.best_score_:.4f}")
    return busqueda.best_estimator_

def metricas(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
    }

def comparar(modelo_nb, X_test, y_test):
    if not os.path.exists(RUTA_SVM):
        print("\nNo se encontró el modelo SVM. Ejecuta primero modelo_svm.py")
        return
    modelo_svm = joblib.load(RUTA_SVM)
    m_svm = metricas(modelo_svm, X_test, y_test)
    m_nb = metricas(modelo_nb, X_test, y_test)

    print("\nCOMPARACIÓN SVM vs NAIVE BAYES (mismos datos de prueba)*")
    print(f"{'Métrica':<20}{'SVM':>10}{'Naive Bayes':>15}")
    for k in m_svm:
        print(f"{k:<20}{m_svm[k]:>10.4f}{m_nb[k]:>15.4f}")

    mejor = "SVM" if m_svm["F1 (macro)"] >= m_nb["F1 (macro)"] else "Naive Bayes"
    print(f"\nMejor modelo según F1 macro: {mejor}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = cargar_datos_y_dividir()
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    modelo = entrenar_nb(X_train, y_train)
    evaluar(modelo, X_test, y_test, "Naive Bayes")
    os.makedirs(os.path.dirname(RUTA_MODELO_NB), exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO_NB)
    print(f"\nModelo guardado en: {RUTA_MODELO_NB}")
    comparar(modelo, X_test, y_test)
