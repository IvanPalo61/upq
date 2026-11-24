import joblib
svm = joblib.load("modelos/modelo_svm.joblib")
nb = joblib.load("modelos/modelo_nb.joblib")

print("Escribe una pregunta (o 'salir' para terminar)")
while True:
    pregunta = input("\nTú: ").strip()
    if pregunta.lower() == "salir":
        break
    texto = [pregunta.lower()]

    pred_svm = svm.predict(texto)[0]
    probs = nb.predict_proba(texto)[0]
    pred_nb = nb.classes_[probs.argmax()]

    print(f"SVM:         {pred_svm}")
    print(f"Naive Bayes: {pred_nb} (confianza {probs.max():.0%})")
    if pred_svm != pred_nb:
        print("  → Los modelos NO coinciden")
