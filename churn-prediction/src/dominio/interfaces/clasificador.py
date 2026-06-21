"""
Módulo de Interfaz: Clasificador
--------------------------------
Este módulo define la interfaz abstracta que establece el contrato
obligatorio para cualquier algoritmo de aprendizaje automático
que se integre en la arquitectura del sistema.
"""

from abc import ABC, abstractmethod
import numpy as np

class Clasificador(ABC):
    """
    Clase base abstracta (Interfaz) para modelos de clasificación supervisada.
    
    Asegura que todos los modelos concretos (como Naive Bayes o Random Forest)
    expongan los mismos métodos básicos para entrenar y predecir, permitiendo a la 
    capa de aplicación utilizarlos de manera estandarizada e intercambiable.
    """

    @abstractmethod
    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo matemático utilizando los datos de entrenamiento proporcionados.

        Args:
            X (np.ndarray): Matriz bidimensional de características de entrenamiento.
            y (np.ndarray): Vector unidimensional de etiquetas objetivo de entrenamiento.
        """
        pass

    @abstractmethod
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula y devuelve las etiquetas discretas predichas para nuevas muestras.

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las clases predichas (ej. 0 o 1) 
                para cada muestra.
        """
        pass

    @abstractmethod
    def predecir_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad estimada de que las muestras pertenezcan a la clase positiva.

        Args:
            X (np.ndarray): Matriz bidimensional de características a evaluar.

        Returns:
            np.ndarray: Vector unidimensional con las probabilidades continuas 
                (entre 0.0 y 1.0) de pertenecer a la clase objetivo (fuga).
        
        Raises:
            NotImplementedError: Si el algoritmo subyacente no soporta matemáticamente
                la estimación de probabilidades.
        """
        pass