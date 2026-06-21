"""
Módulo del Caso de Uso: Evaluar y Comparar Clasificadores
--------------------------------------------------------
Este módulo orquesta la capa de aplicación encargada de coordinar el flujo
de entrenamiento, predicción y cálculo de métricas para múltiples modelos
de Machine Learning, de forma agnóstica a la infraestructura técnica.
"""

from src.dominio.interfaces.clasificador import Clasificador
from src.dominio.entidades.dataset import Dataset
from src.dominio.entidades.metricas import Metricas

class EvaluarYCompararUC:
    """
    Caso de Uso encargado de entrenar, evaluar y comparar un conjunto de clasificadores.
    
    Esta clase implementa la lógica de aplicación que interactúa con las entidades
    e interfaces del dominio para automatizar el procesamiento de los modelos
    de manera unificada.
    """

    def __init__(self, clasificadores: dict[str, Clasificador]) -> None:
        """
        Inicializa el caso de uso con un diccionario de clasificadores disponibles.

        Args:
            clasificadores (dict[str, Clasificador]): Un diccionario donde la clave es el
                nombre del modelo y el valor es una instancia que implementa la interfaz Clasificador.
        """
        self._clasificadores = clasificadores

    def ejecutar(self, datos: Dataset) -> dict[str, Metricas]:
        """
        Ejecuta secuencialmente la canalización de entrenamiento y evaluación
        para cada uno de los clasificadores registrados en el sistema.

        Args:
            datos (Dataset): Entidad de dominio que contiene las matrices particionadas
                para los conjuntos de entrenamiento y prueba.

        Returns:
            dict[str, Metricas]: Un diccionario con las métricas consolidadas por cada
                modelo evaluado, listas para su análisis comparativo o despliegue.
        """
        resultados: dict[str, Metricas] = {}
        for nombre, clf in self._clasificadores.items():
            clf.entrenar(datos.X_entrena, datos.y_entrena)
            y_pred = clf.predecir(datos.X_prueba)
            
            # Intentamos obtener probabilidades para el AUC-ROC
            try:
                y_proba = clf.predecir_proba(datos.X_prueba)
            except NotImplementedError:
                y_proba = None
                
            resultados[nombre] = Metricas.calcular(datos.y_prueba, y_pred, y_proba)
        return resultados