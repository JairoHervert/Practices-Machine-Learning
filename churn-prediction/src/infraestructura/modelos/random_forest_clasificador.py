"""
Módulo de Infraestructura: Clasificador Random Forest
-----------------------------------------------------
Este módulo implementa el adaptador concreto para el algoritmo Random Forest
utilizando scikit-learn. Actúa como el modelo de ensemble propuesto en los
requerimientos del proyecto, destacando por su robustez y su capacidad
para proveer interpretabilidad de negocio.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.dominio.interfaces.clasificador import Clasificador

class RandomForestClasificador(Clasificador):
    """
    Adaptador de RandomForest de scikit-learn al contrato del dominio.
    
    Esta clase envuelve la implementación de Random Forest para cumplir con
    la interfaz Clasificador. Además de los métodos estándar, expone la
    funcionalidad para extraer la importancia de las características.
    """
    
    def __init__(self, semilla: int = 42, n_arboles: int = 200) -> None:
        """
        Inicializa el adaptador configurando los hiperparámetros del modelo.

        Args:
            semilla (int, opcional): Semilla aleatoria para garantizar la
                reproducibilidad del entrenamiento. Por defecto es 42.
            n_arboles (int, opcional): Número de árboles de decisión en el
                bosque (n_estimators). Por defecto es 200.
        """
        self._modelo = RandomForestClassifier(
            n_estimators=n_arboles, 
            random_state=semilla
        )

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el ensamble de árboles a los datos de entrenamiento.

        Args:
            X (np.ndarray): Matriz bidimensional de características estandarizadas.
            y (np.ndarray): Vector unidimensional de etiquetas objetivo.
        """
        self._modelo.fit(X, y)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula las etiquetas predichas por votación mayoritaria del bosque.

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las clases predichas (0 o 1).
        """
        return self._modelo.predict(X)

    def predecir_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad media de las predicciones de los árboles.

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las probabilidades continuas 
                (entre 0.0 y 1.0) correspondientes a la clase positiva (fuga).
        """
        return self._modelo.predict_proba(X)[:, 1]

    def importancia(self) -> np.ndarray:
        """
        Extrae la importancia relativa de cada característica (feature importance).
        
        Este método es fundamental para cumplir con el requerimiento de
        interpretabilidad de negocio, explicando qué factores influyen más
        en la decisión del modelo.

        Returns:
            np.ndarray: Vector unidimensional con los valores de importancia,
                donde la suma de todos los elementos es 1.0.
        """
        return self._modelo.feature_importances_