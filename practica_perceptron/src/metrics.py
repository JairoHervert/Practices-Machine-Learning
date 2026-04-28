"""
Módulo con métricas de evaluación.

Estas funciones se implementan manualmente para evitar depender de sklearn.
Además, se podrán reutilizar tanto en el perceptrón con escalón como en la
neurona sigmoidal con gradiente descendente.
"""

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula la exactitud del modelo.

    La exactitud se define como:

        accuracy = predicciones_correctas / total_de_muestras

    Parámetros
    ----------
    y_true:
        Etiquetas reales.

    y_pred:
        Etiquetas predichas por el modelo.

    Retorna
    -------
    accuracy:
        Valor entre 0 y 1.
    """

    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio.

    Aunque en esta primera implementación el perceptrón produce salidas 0 o 1,
    esta función se deja aquí porque será útil para la implementación sigmoidal.

    Fórmula:

        MSE = promedio((y_real - y_predicho)^2)
    """

    return float(np.mean((y_true - y_pred) ** 2))


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula la matriz de confusión para clasificación binaria.

    En esta práctica se usa:

        Clase positiva: 1 = Maligno
        Clase negativa: 0 = Benigno

    Retorna un diccionario con:

        TN: verdaderos negativos
        FP: falsos positivos
        FN: falsos negativos
        TP: verdaderos positivos
    """

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

    return {
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
    }