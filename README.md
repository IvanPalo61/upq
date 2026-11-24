## Modelos de IA

El chatbot de la UPQ recibe una pregunta y debe decidir a cuál de 15 intenciones pertenece (por ejemplo `becas`, `admision` o `ubicacion`). Para lograrlo se entrenaron dos algoritmos de aprendizaje supervisado con un dataset de preguntas etiquetadas: **SVM** y **Multinomial Naive Bayes**.

### Vectorización TF-IDF

Los algoritmos solo entienden números, así que cada pregunta se convierte en un vector con TF-IDF. Este método da más peso a las palabras que distinguen una intención de otra (como "beca" o "estadía") y menos a las que aparecen en todas partes (como "la" o "de"). Se normalizaron acentos y mayúsculas, y se probaron unigramas y bigramas.

### Modelo 1: SVM (`modelo_svm.py`)

La Máquina de Vectores de Soporte representa cada pregunta como un punto en un espacio de muchas dimensiones y traza fronteras que separan las intenciones, buscando el **mayor margen posible** entre ellas. Se usó la variante lineal (`LinearSVC`), adecuada para texto. Su parámetro principal es `C`, que controla qué tan estricto es el modelo; con GridSearchCV se encontró que `C = 1` funcionaba mejor.

### Modelo 2: Naive Bayes (`modelo_nb.py`)

Este modelo se basa en probabilidades: calcula qué tan probable es que las palabras de la pregunta pertenezcan a cada intención y elige la más probable. Se le llama "ingenuo" porque supone que las palabras son independientes entre sí, una simplificación que lo hace muy rápido, aunque menos preciso cuando varias intenciones comparten vocabulario. Su parámetro principal es `alpha`, un suavizado que evita probabilidades en cero para palabras no vistas.

### Comparación

| Aspecto | SVM | Naive Bayes |
|---|---|---|
| Enfoque | Geométrico: frontera de mayor margen | Probabilístico: intención más probable |
| Supuesto principal | Las clases se separan con fronteras lineales | Las palabras son independientes |
| Parámetro ajustado | `C` | `alpha` |
| Velocidad | Algo más lento | Muy rápido |
| Palabras compartidas entre intenciones | Maneja mejor la ambigüedad | Se confunde más |

### Resultados

Ambos modelos usan la misma división de datos (80 % entrenamiento y 20 % prueba, `random_state = 42`) para que la comparación sea justa.

| Dataset | SVM (accuracy) | Naive Bayes (accuracy) |
|---|---|---|
| 300 preguntas | 0.83 | 0.65 |
| 334 preguntas | 0.75 | 0.57 |

El **SVM obtuvo mejores resultados** en ambos casos. La baja entre un dataset y otro se debe a que el conjunto de prueba cambió y ahora incluye más preguntas de las intenciones difíciles, no a que el modelo empeorara. Las confusiones más frecuentes ocurren entre `admision` y `costo_admision`, que comparten vocabulario.

### Limitaciones

- Ninguno de los dos modelos sabe decir "no sé": ante una pregunta fuera de tema, como "¿Cuál es la capital de Francia?", eligen igualmente una de las 15 intenciones.
- El desempeño depende de la cantidad y variedad de preguntas por intención.

### Cómo ejecutarlo

1. Instalar dependencias: `pip install scikit-learn pandas joblib`
2. Entrenar el SVM: `python modelo_svm.py`
3. Entrenar Naive Bayes y comparar: `python modelo_nb.py`
4. Probar con preguntas propias: `python pruebas_comparadas.py`
