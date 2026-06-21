"""
Módulo de Infraestructura: Clasificador Naive Bayes
---------------------------------------------------
Este módulo implementa el adaptador concreto para el algoritmo Naive Bayes
utilizando la biblioteca scikit-learn. Actúa como el modelo de referencia
(baseline) establecido en los requerimientos del proyecto.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from src.dominio.interfaces.clasificador import Clasificador

class NaiveBayesClasificador(Clasificador):
    """
    Adaptador de GaussianNB de scikit-learn al contrato del dominio.
    
    Esta clase envuelve la implementación de Naive Bayes Gaussiano de sklearn
    para que cumpla con la interfaz abstracta Clasificador, permitiendo su uso
    transparente dentro de la capa de aplicación sin acoplar el dominio a la librería.
    """
    
    def __init__(self):
        """
        Inicializa el adaptador instanciando el modelo GaussianNB subyacente.
        """
        self._modelo = GaussianNB()

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo Naive Bayes a los datos de entrenamiento proporcionados.

        Args:
            X (np.ndarray): Matriz bidimensional de características estandarizadas 
                para el entrenamiento.
            y (np.ndarray): Vector unidimensional de etiquetas objetivo.
        """
        self._modelo.fit(X, y)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula las etiquetas predichas para las muestras proporcionadas.

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las clases predichas (0 o 1).
        """
        return self._modelo.predict(X)

    def predecir_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad estimada de pertenecer a la clase positiva (fuga).

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las probabilidades continuas 
                (entre 0.0 y 1.0) correspondientes a la clase 1.
        """
        return self._modelo.predict_proba(X)[:, 1]