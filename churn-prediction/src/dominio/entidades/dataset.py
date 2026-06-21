"""
Módulo de Entidad: Dataset
--------------------------
Este módulo define la estructura de datos principal de la capa de dominio
para almacenar y transferir los conjuntos de entrenamiento y prueba
entre los distintos componentes de la arquitectura.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class Dataset:
    """
    Entidad de dominio que representa un conjunto de datos particionado.
    
    Agrupa las matrices de características (X) y los vectores de etiquetas (y)
    necesarios para el entrenamiento y evaluación de los modelos de Machine Learning,
    garantizando su integridad durante el flujo de la aplicación.
    
    Atributos:
        X_entrena (np.ndarray): Matriz de características utilizada para el entrenamiento.
        X_prueba (np.ndarray): Matriz de características utilizada para la validación o prueba.
        y_entrena (np.ndarray): Vector de etiquetas objetivo correspondientes al entrenamiento.
        y_prueba (np.ndarray): Vector de etiquetas objetivo correspondientes a la prueba.
    """
    X_entrena: np.ndarray
    X_prueba: np.ndarray
    y_entrena: np.ndarray
    y_prueba: np.ndarray