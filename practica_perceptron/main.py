"""
Archivo principal de ejecución de la práctica.

Este archivo ejecuta las dos primeras implementaciones:

    1. Perceptrón con función escalón + regla delta.
    2. Neurona sigmoidal + gradiente descendente.

Flujo general:
    1. Cargar el dataset local data/wdbc.data.
    2. Dividir en entrenamiento y prueba.
    3. Estandarizar las características.
    4. Entrenar ambos modelos.
    5. Evaluar ambos modelos con accuracy, MSE y matriz de confusión.

El dataset no se carga desde sklearn. Se lee directamente desde el archivo local
wdbc.data, como solicitó el profesor.
"""

from pathlib import Path

from src.data_loader import (
    StandardScalerScratch,
    load_wdbc_dataset,
    stratified_train_test_split,
)
from src.metrics import accuracy_score, confusion_matrix_binary, mean_squared_error
from src.perceptron import PerceptronDelta
from src.sigmoid_gradient import SigmoidGradientClassifier


DATA_PATH = Path("data/wdbc.data")


def print_evaluation_report(model_name: str, y_true, y_pred, y_score=None) -> None:
    """
    Imprime un reporte simple de evaluación.

    Parámetros
    ----------
    model_name:
        Nombre del modelo evaluado.

    y_true:
        Etiquetas reales.

    y_pred:
        Etiquetas predichas en formato 0 o 1.

    y_score:
        Salidas continuas del modelo. Este parámetro es opcional.

    Notas
    -----
    En el perceptrón con escalón, la salida ya es 0 o 1, por lo que el MSE se
    calcula con y_pred.

    En la neurona sigmoidal, la salida natural del modelo es un valor continuo
    entre 0 y 1. Por eso, para el MSE conviene usar y_score.
    """

    accuracy = accuracy_score(y_true, y_pred)

    if y_score is None:
        mse = mean_squared_error(y_true, y_pred)
    else:
        mse = mean_squared_error(y_true, y_score)

    confusion_matrix = confusion_matrix_binary(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MSE: {mse:.6f}")

    print("\nMatriz de confusión:")
    print(f"TN: {confusion_matrix['TN']} | FP: {confusion_matrix['FP']}")
    print(f"FN: {confusion_matrix['FN']} | TP: {confusion_matrix['TP']}")


def print_training_history(model_name: str, history: list[dict], last_n: int = 10) -> None:
    """
    Imprime las últimas épocas del entrenamiento.

    Esta función está preparada para mostrar el historial de ambos modelos.

    En el perceptrón existe la clave 'errors', porque se cuentan los errores
    cometidos en cada época.

    En la neurona sigmoidal no se usa esa clave, porque se optimiza una salida
    continua mediante gradiente.
    """

    print(f"\nÚltimas {last_n} épocas de entrenamiento - {model_name}:")

    for row in history[-last_n:]:
        message = (
            f"Época {row['epoch']:04d} | "
            f"Accuracy: {row['accuracy']:.4f} | "
            f"MSE: {row['mse']:.6f}"
        )

        # El perceptrón guarda errores por época; el sigmoidal no.
        if "errors" in row:
            message = (
                f"Época {row['epoch']:04d} | "
                f"Errores: {row['errors']:03d} | "
                f"Accuracy: {row['accuracy']:.4f} | "
                f"MSE: {row['mse']:.6f}"
            )

        print(message)


def run_perceptron(X_train, X_test, y_train, y_test) -> None:
    """
    Ejecuta la implementación 1:
    Perceptrón con función escalón + regla delta.
    """

    perceptron = PerceptronDelta(
        learning_rate=0.01,
        n_epochs=100,
        random_state=42,
        shuffle=True,
    )

    print("\nEntrenando perceptrón con escalón + regla delta...")
    perceptron.fit(X_train, y_train)

    train_predictions = perceptron.predict(X_train)
    test_predictions = perceptron.predict(X_test)

    print_training_history(
        model_name="Perceptrón escalón + regla delta",
        history=perceptron.history_,
    )

    print_evaluation_report(
        model_name="Perceptrón escalón + regla delta [entrenamiento]",
        y_true=y_train,
        y_pred=train_predictions,
    )

    print_evaluation_report(
        model_name="Perceptrón escalón + regla delta [prueba]",
        y_true=y_test,
        y_pred=test_predictions,
    )


def run_sigmoid_gradient(X_train, X_test, y_train, y_test) -> None:
    """
    Ejecuta la implementación 2:
    Neurona sigmoidal + gradiente descendente.
    """

    sigmoid_model = SigmoidGradientClassifier(
        learning_rate=0.5,
        n_epochs=1500,
        random_state=42,
        threshold=0.5,
    )

    print("\nEntrenando neurona sigmoidal + gradiente descendente...")
    sigmoid_model.fit(X_train, y_train)

    train_probabilities = sigmoid_model.predict_proba(X_train)
    test_probabilities = sigmoid_model.predict_proba(X_test)

    train_predictions = sigmoid_model.predict(X_train)
    test_predictions = sigmoid_model.predict(X_test)

    print_training_history(
        model_name="Sigmoide + gradiente descendente",
        history=sigmoid_model.history_,
    )

    print_evaluation_report(
        model_name="Sigmoide + gradiente [entrenamiento]",
        y_true=y_train,
        y_pred=train_predictions,
        y_score=train_probabilities,
    )

    print_evaluation_report(
        model_name="Sigmoide + gradiente [prueba]",
        y_true=y_test,
        y_pred=test_predictions,
        y_score=test_probabilities,
    )


def main() -> None:
    """
    Punto de entrada del programa.
    """

    print("Cargando dataset WDBC desde archivo local...")

    X, y = load_wdbc_dataset(DATA_PATH)

    print(f"Muestras totales: {X.shape[0]}")
    print(f"Características: {X.shape[1]}")

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"Muestras de prueba: {X_test.shape[0]}")

    scaler = StandardScalerScratch()    # z = (x - media) / desviación estándar

    # El escalador se ajusta únicamente con entrenamiento.
    X_train_scaled = scaler.fit_transform(X_train)

    # El conjunto de prueba se transforma con la misma media y desviación.
    X_test_scaled = scaler.transform(X_test)

    run_perceptron(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )

    run_sigmoid_gradient(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()