
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.metrics import accuracy_score


def stratified_k_fold_indices(
    y: pd.Series | list[Any] | np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Genera índices de entrenamiento y prueba usando Stratified K-Fold.

    Esta implementación es manual y está pensada con fines didácticos:
    preserva, en la medida de lo posible, la proporción de clases
    en cada fold.

    Parámetros:
    y : etiquetas de clase.
    n_splits : int
        Número de particiones.
    shuffle : bool
        Indica si se barajan los índices de cada clase.
    random_state : int
        Semilla para reproducibilidad.

    Yields:
    (train_idx, test_idx)
    """
    y_list = list(y)

    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")

    class_indices = defaultdict(list)
    for idx, label in enumerate(y_list):
        class_indices[label].append(idx)

    min_class_count = min(len(indices) for indices in class_indices.values())
    if n_splits > min_class_count:
        raise ValueError(
            "n_splits no puede ser mayor que el número de elementos "
            "de la clase menos frecuente."
        )

    rng = np.random.default_rng(random_state)

    fold_test_indices = [[] for _ in range(n_splits)]

    for label, indices in class_indices.items():
        indices = np.array(indices, dtype=int)
        if shuffle:
            rng.shuffle(indices)

        parts = np.array_split(indices, n_splits)
        for fold_id, part in enumerate(parts):
            fold_test_indices[fold_id].extend(part.tolist())

    all_indices = set(range(len(y_list)))

    for fold_id in range(n_splits):
        test_idx = np.array(sorted(fold_test_indices[fold_id]), dtype=int)
        train_idx = np.array(sorted(all_indices - set(test_idx.tolist())), dtype=int)
        yield train_idx, test_idx


def cross_validate_classifier(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Evalúa un clasificador mediante validación cruzada estratificada.

    El clasificador debe exponer una interfaz mínima:
    - fit(X_train, y_train)
    - predict(X_test)

    Retorna:
    list[dict]
        Lista de resultados por fold.
    """
    results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        stratified_k_fold_indices(
            y=y,
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ),
        start=1,
    ):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)

        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results.append(
            {
                "fold": fold_number,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "accuracy": acc,
                "y_true": y_test.tolist(),
                "y_pred": list(y_pred),
            }
        )

    return results


def summarize_cross_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Resume los resultados obtenidos en validación cruzada.
    """
    accuracies = [item["accuracy"] for item in results]
    accuracies_np = np.array(accuracies, dtype=float)

    return {
        "fold_accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies_np)),
        "std_accuracy": float(np.std(accuracies_np, ddof=0)),
        "min_accuracy": float(np.min(accuracies_np)),
        "max_accuracy": float(np.max(accuracies_np)),
    }
