
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


class KNNClassifier:
    """
    Implementa un clasificador K-Nearest Neighbors (KNN) para problemas
    de clasificación multiclase.

    Características principales:
    - Usa distancia euclidiana.
    - Usa voto mayoritario simple (sin peso).
    - Permite devolver detalles de la predicción para fines didácticos.
    - Trabaja con pandas.DataFrame, numpy.ndarray o listas anidadas.

    Este modelo es un clasificador "perezoso":
    no construye una estructura probabilística como Naive Bayes,
    sino que memoriza el conjunto de entrenamiento y decide la clase
    de una nueva muestra observando a sus vecinos más cercanos.
    """

    def __init__(self, k: int = 5) -> None:
        """
        Inicializa el clasificador.

        Parámetros:
        k : int
            Número de vecinos más cercanos que participarán
            en la votación.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k debe ser un entero positivo.")

        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.classes_: list[Any] = []

    def _to_numpy_features(self, X: pd.DataFrame | np.ndarray | list[list[float]]) -> np.ndarray:
        """
        Convierte las características de entrada a un arreglo de NumPy
        de tipo float.

        Esto unifica el manejo interno del modelo y evita depender
        del formato exacto en que el usuario entrega los datos.
        """
        if isinstance(X, pd.DataFrame):
            if not self.feature_names:
                self.feature_names = list(X.columns)
            return X.astype(float).to_numpy()

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def fit(self, X: pd.DataFrame | np.ndarray | list[list[float]], y: Iterable[Any]) -> "KNNClassifier":
        """
        Almacena el conjunto de entrenamiento.

        Parámetros:
        X : matriz de características.
        y : etiquetas de clase asociadas a cada fila de X.

        Retorna:
        self
        """
        X_array = self._to_numpy_features(X)
        y_array = np.asarray(list(y))

        if len(X_array) != len(y_array):
            raise ValueError("X y y deben tener el mismo número de filas.")

        if self.k > len(X_array):
            raise ValueError(
                f"k={self.k} no puede ser mayor al número de muestras "
                f"de entrenamiento ({len(X_array)})."
            )

        self.X_train = X_array
        self.y_train = y_array
        self.classes_ = sorted(pd.unique(y_array).tolist(), key=str)
        return self

    def _euclidean_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclidiana entre una muestra nueva
        y todas las muestras del entrenamiento.
        """
        if self.X_train is None:
            raise ValueError("El modelo aún no ha sido entrenado.")

        diff = self.X_train - sample
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _resolve_tie(
        self,
        neighbors_labels: np.ndarray,
        neighbors_distances: np.ndarray,
    ) -> Any:
        """
        Resuelve empates de forma determinista.

        Criterio:
        1. Se eligen las clases con mayor número de votos.
        2. Si hay empate, gana la clase con menor suma de distancias
           entre sus vecinos seleccionados.
        3. Si el empate persiste, se usa el orden alfabético de la clase
           para garantizar reproducibilidad.
        """
        votes = Counter(neighbors_labels)
        max_votes = max(votes.values())
        candidates = [label for label, count in votes.items() if count == max_votes]

        if len(candidates) == 1:
            return candidates[0]

        distance_sum = defaultdict(float)
        for label, distance in zip(neighbors_labels, neighbors_distances):
            if label in candidates:
                distance_sum[label] += float(distance)

        best_distance = min(distance_sum[label] for label in candidates)
        candidates = [label for label in candidates if distance_sum[label] == best_distance]

        return sorted(candidates, key=str)[0]

    def predict_one(
        self,
        sample: dict[str, Any] | pd.Series | np.ndarray | list[float],
        return_details: bool = False,
    ):
        """
        Predice la clase de una sola muestra.

        Parámetros:
        sample : muestra individual.
        return_details : bool
            Si es True, también devuelve información detallada
            de vecinos, distancias y votos.

        Retorna:
        clase_predicha
        o
        (clase_predicha, details)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("El modelo aún no ha sido entrenado.")

        if isinstance(sample, dict):
            if not self.feature_names:
                raise ValueError("No se detectaron nombres de columnas del entrenamiento.")
            sample_array = np.array([float(sample[name]) for name in self.feature_names], dtype=float)
        elif isinstance(sample, pd.Series):
            sample_array = sample.astype(float).to_numpy()
        else:
            sample_array = np.asarray(sample, dtype=float)

        if sample_array.ndim != 1:
            raise ValueError("sample debe representar una sola observación.")

        distances = self._euclidean_distances(sample_array)
        neighbor_idx = np.argsort(distances)[: self.k]
        neighbor_distances = distances[neighbor_idx]
        neighbor_labels = self.y_train[neighbor_idx]

        predicted_class = self._resolve_tie(neighbor_labels, neighbor_distances)

        if return_details:
            vote_counter = Counter(neighbor_labels)
            details = {
                "k": self.k,
                "votes": dict(vote_counter),
                "neighbors": [
                    {
                        "index_train": int(idx),
                        "distance": float(dist),
                        "label": label,
                    }
                    for idx, dist, label in zip(neighbor_idx, neighbor_distances, neighbor_labels)
                ],
            }
            return predicted_class, details

        return predicted_class

    def predict(self, X: pd.DataFrame | np.ndarray | list[list[float]]) -> list[Any]:
        """
        Predice la clase de múltiples muestras.
        """
        X_array = self._to_numpy_features(X)
        return [self.predict_one(row) for row in X_array]

    def summary(self) -> dict[str, Any]:
        """
        Devuelve un resumen breve del modelo entrenado.
        """
        if self.X_train is None or self.y_train is None:
            return {
                "trained": False,
                "k": self.k,
            }

        return {
            "trained": True,
            "k": self.k,
            "n_samples": int(len(self.X_train)),
            "n_features": int(self.X_train.shape[1]),
            "classes": self.classes_,
            "feature_names": self.feature_names,
        }
