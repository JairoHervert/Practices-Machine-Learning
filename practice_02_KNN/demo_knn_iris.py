
from __future__ import annotations

from pathlib import Path

from colorama import Fore, Style, init

from models.knn_classifier import KNNClassifier
from utils.data_loader import load_iris_dataset, split_features_target

init(autoreset=True)

TITLE = Fore.WHITE + Style.BRIGHT
INFO = Fore.BLUE + Style.BRIGHT
OK = Fore.GREEN + Style.BRIGHT
WARN = Fore.YELLOW + Style.BRIGHT
ERR = Fore.RED + Style.BRIGHT
RESET = Style.RESET_ALL


def print_separator() -> None:
    """Imprime una línea divisoria para mejorar la lectura en consola."""
    print(TITLE + "=" * 78)


def ask_float(feature_name: str) -> float:
    """
    Solicita al usuario un valor numérico para una característica.
    """
    while True:
        try:
            return float(input(INFO + f"{feature_name}: " + RESET))
        except ValueError:
            print(ERR + "Entrada inválida. Debes escribir un número.")


def show_result(prediction: str, details: dict) -> None:
    """
    Muestra el resultado de la predicción y el detalle de vecinos.
    """
    print_separator()
    print(TITLE + "RESULTADO FINAL")
    print_separator()
    print(OK + f"Clase predicha: {prediction}")

    print(INFO + "\nDetalle de vecinos usados en la decisión:")
    for i, neighbor in enumerate(details["neighbors"], start=1):
        print(
            f"  Vecino {i}: "
            f"índice_train={neighbor['index_train']}, "
            f"distancia={neighbor['distance']:.4f}, "
            f"clase={neighbor['label']}"
        )

    print(INFO + "\nVotación final:")
    for label, votes in details["votes"].items():
        print(f"  {label}: {votes} voto(s)")


def main() -> None:
    """
    Programa interactivo para probar el clasificador KNN con Iris.

    Flujo general:
    1. Carga el dataset externo descargado.
    2. Entrena KNN con todo el conjunto.
    3. Solicita una flor nueva desde teclado.
    4. Muestra predicción, vecinos y votos.
    """
    dataset_path = Path("data/iris.data")
    df = load_iris_dataset(dataset_path)
    X, y = split_features_target(df, target_col="species")

    model = KNNClassifier(k=5)
    model.fit(X, y)

    print_separator()
    print(TITLE + "KNN SIN PESO - DATASET IRIS")
    print_separator()
    print(INFO + "Resumen del modelo:")
    print(model.summary())

    while True:
        print_separator()
        print(TITLE + "CAPTURA DE NUEVA FLOR")
        print_separator()

        sample = {
            "sepal_length_cm": ask_float("sepal_length_cm"),
            "sepal_width_cm": ask_float("sepal_width_cm"),
            "petal_length_cm": ask_float("petal_length_cm"),
            "petal_width_cm": ask_float("petal_width_cm"),
        }

        prediction, details = model.predict_one(sample, return_details=True)
        show_result(prediction, details)

        again = input(INFO + "\n¿Deseas probar otra flor? (s/n): " + RESET).strip().lower()
        if again != "s":
            print(OK + "\nPrograma terminado.")
            break


if __name__ == "__main__":
    main()
