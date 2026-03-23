
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from models.naive_bayes import NaiveBayes


class GaussianNaiveBayesAdapter:
    """
    Adaptador para reutilizar la implementación de Naive Bayes
    de la práctica anterior dentro de una interfaz uniforme.

    El objetivo es que tanto KNN como Naive Bayes puedan evaluarse
    usando la misma función de validación cruzada.
    """

    def __init__(self, alpha: float = 1.0, target_col: str = "target") -> None:
        self.alpha = alpha
        self.target_col = target_col
        self.model = NaiveBayes(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: Iterable[Any]) -> "GaussianNaiveBayesAdapter":
        """
        Entrena el modelo a partir de X y y separados.
        """
        df = X.copy()
        df[self.target_col] = list(y)
        self.model.fit(df, target_col=self.target_col)
        return self

    def predict(self, X: pd.DataFrame) -> list[Any]:
        """
        Predice las clases de un conjunto de muestras.
        """
        return self.model.predict(X)

    def summary(self) -> dict[str, Any]:
        """
        Devuelve el resumen interno del modelo Naive Bayes.
        """
        return self.model.summary()
