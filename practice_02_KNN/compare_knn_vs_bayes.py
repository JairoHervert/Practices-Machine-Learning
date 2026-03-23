
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.gaussian_nb_adapter import GaussianNaiveBayesAdapter
from models.knn_classifier import KNNClassifier
from utils.data_loader import load_iris_dataset, split_features_target
from utils.validation import cross_validate_classifier, summarize_cross_validation


def print_model_report(model_name: str, results: list[dict], summary: dict) -> None:
    """
    Imprime los resultados por fold y el resumen final de un modelo.
    """
    print("=" * 78)
    print(model_name.upper())
    print("=" * 78)

    rows = []
    for item in results:
        rows.append(
            {
                "Fold": item["fold"],
                "Train": item["n_train"],
                "Test": item["n_test"],
                "Accuracy": round(item["accuracy"], 4),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(f"Accuracy promedio : {summary['mean_accuracy']:.4f}")
    print(f"Desviación estándar: {summary['std_accuracy']:.4f}")
    print(f"Accuracy mínimo    : {summary['min_accuracy']:.4f}")
    print(f"Accuracy máximo    : {summary['max_accuracy']:.4f}")
    print()


def main() -> None:
    """
    Evalúa y compara KNN y Naive Bayes con el mismo esquema
    de validación cruzada estratificada.

    Esto permite una comparación más justa entre ambas prácticas.
    """
    dataset_path = Path("data/iris.data")
    df = load_iris_dataset(dataset_path)
    X, y = split_features_target(df, target_col="species")

    knn_results = cross_validate_classifier(
        model_factory=lambda: KNNClassifier(k=5),
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )
    knn_summary = summarize_cross_validation(knn_results)

    bayes_results = cross_validate_classifier(
        model_factory=lambda: GaussianNaiveBayesAdapter(alpha=1.0, target_col="species"),
        X=X,
        y=y,
        n_splits=5,
        random_state=42,
    )
    bayes_summary = summarize_cross_validation(bayes_results)

    print_model_report("KNN sin peso (k=5)", knn_results, knn_summary)
    print_model_report("Naive Bayes gaussiano", bayes_results, bayes_summary)

    if knn_summary["mean_accuracy"] > bayes_summary["mean_accuracy"]:
        winner = "KNN"
    elif knn_summary["mean_accuracy"] < bayes_summary["mean_accuracy"]:
        winner = "Naive Bayes"
    else:
        winner = "Empate"

    print("=" * 78)
    print("COMPARACIÓN FINAL")
    print("=" * 78)
    print(f"Ganador por accuracy promedio: {winner}")
    print(
        f"Diferencia absoluta: "
        f"{abs(knn_summary['mean_accuracy'] - bayes_summary['mean_accuracy']):.4f}"
    )


if __name__ == "__main__":
    main()
