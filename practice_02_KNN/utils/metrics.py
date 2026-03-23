
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import pandas as pd


def accuracy_score(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    """
    Calcula la exactitud (accuracy).

    Accuracy = número de aciertos / número total de ejemplos.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud.")

    correct = sum(1 for real, pred in zip(y_true, y_pred) if real == pred)
    return correct / len(y_true) if y_true else 0.0


def confusion_matrix(y_true: Iterable[Any], y_pred: Iterable[Any]) -> pd.DataFrame:
    """
    Construye una matriz de confusión simple usando pandas.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(pd.unique(y_true + y_pred).tolist(), key=str)

    matrix = defaultdict(lambda: defaultdict(int))
    for real, pred in zip(y_true, y_pred):
        matrix[real][pred] += 1

    data = []
    for real in labels:
        row = []
        for pred in labels:
            row.append(matrix[real][pred])
        data.append(row)

    return pd.DataFrame(data, index=labels, columns=labels)
