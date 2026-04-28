"""
Módulo encargado de cargar y preparar el dataset WDBC.

Este archivo concentra funciones reutilizables para las dos implementaciones
de la práctica:

1. Perceptrón con función escalón y regla delta.
2. Neurona sigmoidal con gradiente descendente.

El profesor indicó que el dataset no debe cargarse desde sklearn, por lo que
este módulo lee directamente el archivo local data/wdbc.data.

El archivo wdbc.data no contiene encabezados, por lo que los nombres de las
columnas se declaran manualmente de acuerdo con la descripción del dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

COLUMN_NAMES = ["id", "diagnosis", *FEATURE_NAMES]


def load_wdbc_dataset(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga el dataset WDBC desde un archivo local.

    Parámetros
    ----------
    file_path:
        Ruta del archivo wdbc.data.

    Retorna
    -------
    X:
        Matriz de características con forma (n_muestras, n_características).

    y:
        Vector de etiquetas con valores 0 y 1.

    Codificación usada
    ------------------
    B -> 0  Benigno
    M -> 1  Maligno

    Esta codificación se eligió porque el perceptrón binario trabaja de forma
    natural con dos clases numéricas.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}\n"
            "Verifica que wdbc.data esté dentro de la carpeta data/."
        )

    # El archivo original viene separado por comas y no contiene encabezados.
    dataframe = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    # Convertimos las 30 características a valores flotantes.
    X = dataframe[FEATURE_NAMES].to_numpy(dtype=float)

    # Convertimos la etiqueta textual M/B a una etiqueta numérica 1/0.
    y = dataframe["diagnosis"].map({"B": 0, "M": 1}).to_numpy(dtype=int)

    return X, y


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el dataset en entrenamiento y prueba conservando la proporción
    aproximada de clases.

    Esta función se implementa manualmente para evitar depender de sklearn.

    Parámetros
    ----------
    X:
        Matriz de características.

    y:
        Vector de etiquetas.

    test_size:
        Proporción del dataset que se usará para prueba.

    random_state:
        Semilla para que la división sea reproducible.

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")

    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    # Separamos índices por clase para conservar proporciones.
    for class_value in np.unique(y):
        class_indices = np.where(y == class_value)[0]
        rng.shuffle(class_indices)

        n_test = int(round(len(class_indices) * test_size))

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Mezclamos los índices finales para que no queden ordenados por clase.
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class StandardScalerScratch:
    """
    Estandarizador implementado desde cero.

    La estandarización transforma cada característica usando:

        x_estandarizado = (x - media) / desviación_estándar

    Esto es importante porque el perceptrón usa productos punto. Si una
    característica tiene valores mucho más grandes que otra, puede dominar
    el cálculo de la suma ponderada.

    Esta clase se podrá reutilizar también en la implementación sigmoidal.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        """
        Calcula la media y desviación estándar usando solo datos de entrenamiento.
        """

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        # Evita divisiones entre cero si alguna columna tuviera desviación 0.
        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la estandarización usando la media y desviación ya calculadas.
        """

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Primero debes llamar al método fit(X).")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula los parámetros de escalado y transforma los datos.
        """

        return self.fit(X).transform(X)