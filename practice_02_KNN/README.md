
# Práctica 02 - KNN con Iris y comparación contra Naive Bayes

Este proyecto implementa:

1. Un clasificador **KNN sin peso** (voto uniforme).
2. Carga del dataset **Iris** desde un archivo descargado externamente.
3. **Validación cruzada estratificada** para evaluar el modelo.
4. Comparación directa entre **KNN** y **Naive Bayes** usando los mismos folds.
5. Un programa interactivo para capturar una nueva flor desde teclado.

## Estructura

```text
practica_02_knn/
├── data/
│   └── iris.data
├── models/
│   ├── gaussian_nb_adapter.py
│   ├── knn_classifier.py
│   └── naive_bayes.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── validation.py
├── compare_knn_vs_bayes.py
├── demo_knn_iris.py
├── README.md
└── requirements.txt
```

## ¿Qué hace cada archivo?

- `models/knn_classifier.py`
  Implementa KNN desde cero usando distancia euclidiana y voto mayoritario simple.

- `models/naive_bayes.py`
  Reutiliza la implementación de la práctica anterior.

- `models/gaussian_nb_adapter.py`
  Adapta Naive Bayes para evaluarlo con la misma interfaz que KNN.

- `utils/data_loader.py`
  Carga el archivo `iris.data` descargado externamente y lo convierte a DataFrame.

- `utils/validation.py`
  Implementa una versión manual de **Stratified K-Fold** y la evaluación por folds.

- `compare_knn_vs_bayes.py`
  Evalúa ambos modelos y muestra sus accuracies por fold y su promedio final.

- `demo_knn_iris.py`
  Programa interactivo para ingresar una flor nueva y obtener su clase.

## Dependencias

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta del proyecto:

```bash
python compare_knn_vs_bayes.py
```

Para la demo interactiva:

```bash
python demo_knn_iris.py
```

## Comentario metodológico

Para comparar correctamente con Naive Bayes,
ambos modelos se evalúan sobre el mismo dataset
y con exactamente los mismos folds de validación.
