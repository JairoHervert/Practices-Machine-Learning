"""
Módulo de Entidad: Métricas
---------------------------
Este módulo define la entidad encargada de representar y calcular
las métricas de evaluación de los modelos de clasificación. Centraliza
la medición del rendimiento para garantizar consistencia en los reportes.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@dataclass
class Metricas:
    """
    Entidad de dominio que consolida los resultados de la evaluación de un modelo.
    
    Atributos:
        exactitud (float): Proporción de predicciones correctas sobre el total (Accuracy).
        precision (float): Proporción de verdaderos positivos sobre las predicciones positivas (Precision).
        exhaustividad (float): Proporción de verdaderos positivos sobre los positivos reales (Recall).
        f1 (float): Media armónica entre la precisión y la exhaustividad (F1-Score).
        auc_roc (float): Área bajo la curva Característica Operativa del Receptor (AUC-ROC).
    """
    exactitud: float
    precision: float
    exhaustividad: float
    f1: float
    auc_roc: float

    @classmethod
    def calcular(cls, y_verdadera: np.ndarray, y_predicha: np.ndarray, y_proba: np.ndarray = None) -> 'Metricas':
        """
        Calcula las métricas de clasificación a partir de las predicciones del modelo.

        Args:
            y_verdadera (np.ndarray): Arreglo con las etiquetas reales (0 o 1).
            y_predicha (np.ndarray): Arreglo con las clases predichas por el modelo.
            y_proba (np.ndarray, opcional): Arreglo con las probabilidades para la clase positiva. 
                Es necesario para calcular el AUC-ROC; si se omite, se asigna 0.0.

        Returns:
            Metricas: Una nueva instancia con los valores de rendimiento calculados.
        """
        # AUC-ROC requiere probabilidades. Si no las tenemos, ponemos 0.0
        auc = roc_auc_score(y_verdadera, y_proba) if y_proba is not None else 0.0
        return cls(
            exactitud=accuracy_score(y_verdadera, y_predicha),
            precision=precision_score(y_verdadera, y_predicha, zero_division=0),
            exhaustividad=recall_score(y_verdadera, y_predicha, zero_division=0),
            f1=f1_score(y_verdadera, y_predicha, zero_division=0),
            auc_roc=auc
        )