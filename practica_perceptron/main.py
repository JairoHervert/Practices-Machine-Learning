"""
Archivo principal de ejecución de la práctica.

Este archivo ejecuta las tres implementaciones:

    1. Perceptrón con función escalón + regla delta.
    2. Neurona sigmoidal + gradiente descendente.
    3. Neurona sigmoidal + PSO (Versión NumPy y Versión Manual).

Flujo general:
    1. Cargar el dataset local data/wdbc.data.
    2. Dividir en entrenamiento y prueba.
    3. Estandarizar las características.
    4. Entrenar los modelos y medir tiempos.
    5. Evaluar los modelos con accuracy, MSE y matriz de confusión.
"""

from pathlib import Path
import numpy as np
import time

from src.data_loader import (
    StandardScalerScratch,
    load_wdbc_dataset,
    stratified_train_test_split,
)
from src.metrics import accuracy_score, confusion_matrix_binary, mean_squared_error
from src.perceptron import PerceptronDelta
from src.sigmoid_gradient import SigmoidGradientClassifier
from src.sigmoid_pso import SigmoidPSOClassifier
from src.sigmoid_pso_manual import SigmoidPSOManual


DATA_PATH = Path("data/wdbc.data")


def print_evaluation_report(model_name: str, y_true, y_pred, y_score=None) -> None:
    """
    Imprime un reporte simple de evaluación.
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
    """

    print(f"\nÚltimas {last_n} épocas de entrenamiento - {model_name}:")

    for row in history[-last_n:]:
        message = (
            f"Época {row['epoch']:04d} | "
            f"Accuracy: {row['accuracy']:.4f} | "
            f"MSE: {row['mse']:.6f}"
        )

        if "errors" in row:
            message = (
                f"Época {row['epoch']:04d} | "
                f"Errores: {row['errors']:03d} | "
                f"Accuracy: {row['accuracy']:.4f} | "
                f"MSE: {row['mse']:.6f}"
            )
        
        if "error_sum" in row:
             message += f" | Error Total: {row['error_sum']:.4f}"

        print(message)


def run_perceptron(X_train, X_test, y_train, y_test) -> tuple[float, float, float]:
    """
    Ejecuta la implementación 1: Perceptrón Delta.
    """
    perceptron = PerceptronDelta(learning_rate=0.01, n_epochs=100, random_state=42)
    
    print("\nEntrenando perceptrón con escalón + regla delta...")
    start = time.time()
    perceptron.fit(X_train, y_train)
    t_exec = time.time() - start

    train_pred = perceptron.predict(X_train)
    test_pred = perceptron.predict(X_test)

    print_evaluation_report("Perceptrón Delta [prueba]", y_test, test_pred)

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred), t_exec


def run_sigmoid_gradient(X_train, X_test, y_train, y_test) -> tuple[float, float, float]:
    """
    Ejecuta la implementación 2: Gradiente Descendente.
    """
    model = SigmoidGradientClassifier(learning_rate=0.5, n_epochs=1500, random_state=42)
    
    print("\nEntrenando neurona sigmoidal + gradiente descendente...")
    start = time.time()
    model.fit(X_train, y_train)
    t_exec = time.time() - start

    train_pred = np.array(model.predict(X_train))
    test_pred = np.array(model.predict(X_test))

    print_evaluation_report("Sigmoide + Gradiente [prueba]", y_test, test_pred)

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred), t_exec


def run_sigmoid_pso(X_train, X_test, y_train, y_test) -> tuple[float, float, float]:
    """
    Ejecuta la implementación 3: PSO Vectorizado.
    """
    model = SigmoidPSOClassifier(n_particles=40, n_iterations=100, random_state=42)
    
    print("\nEntrenando neurona sigmoidal + PSO (NumPy Vectorizado)...")
    start = time.time()
    model.fit(X_train, y_train)
    t_exec = time.time() - start

    train_pred = np.array(model.predict(X_train))
    test_pred = np.array(model.predict(X_test))

    print_evaluation_report("Sigmoide + PSO Vectorizado [prueba]", y_test, test_pred)

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred), t_exec


def run_sigmoid_pso_manual(X_train, X_test, y_train, y_test) -> tuple[float, float, float]:
    """
    Ejecuta la implementación 3 (Manual): PSO sin NumPy.
    """
    model = SigmoidPSOManual(n_particles=40, n_iterations=100, random_state=42)
    
    print("\nEntrenando neurona sigmoidal + PSO (MANUAL - BUCLAS EXPLÍCITOS)...")
    print("Nota: Esta versión es deliberadamente lenta para fines educativos.")
    model.fit(X_train, y_train)
    t_exec = model.training_time

    train_pred = np.array(model.predict(X_train))
    test_pred = np.array(model.predict(X_test))

    print_evaluation_report("Sigmoide + PSO Manual [prueba]", y_test, test_pred)

    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred), t_exec


def print_comparison_table(results: dict) -> None:
    """
    Imprime una tabla comparativa final.
    """
    print("\n" + "="*85)
    print(f"{'TABLA COMPARATIVA FINAL DE RENDIMIENTO':^85}")
    print("="*85)
    print(f"{'Modelo':<35} | {'Acc. Train':<12} | {'Acc. Test':<10} | {'Tiempo (s)':<12}")
    print("-"*85)

    for name, (tr_acc, ts_acc, t_ex) in results.items():
        print(f"{name:<35} | {tr_acc:^12.4f} | {ts_acc:^10.4f} | {t_ex:^12.4f}")

    print("="*85)


def main() -> None:
    X, y = load_wdbc_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    results["Perceptrón Delta"] = run_perceptron(X_train_scaled, X_test_scaled, y_train, y_test)
    results["Sigmoide (Gradiente)"] = run_sigmoid_gradient(X_train_scaled, X_test_scaled, y_train, y_test)
    results["Sigmoide (PSO Manual)"] = run_sigmoid_pso_manual(X_train_scaled, X_test_scaled, y_train, y_test)
    results["Sigmoide (PSO Vectorizado)"] = run_sigmoid_pso(X_train_scaled, X_test_scaled, y_train, y_test)

    print_comparison_table(results)


if __name__ == "__main__":
    main()
