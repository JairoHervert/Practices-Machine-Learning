"""
Módulo de Interfaz: Cargador de Datos
------------------------------------
Este módulo define la interfaz abstracta que actúa como contrato de dominio
para las operaciones de lectura, limpieza y partición de datos, abstrayendo
la lógica del origen físico de la información (CSV, base de datos, etc.).
"""

from abc import ABC, abstractmethod
from src.dominio.entidades.dataset import Dataset

class CargadorDatos(ABC):
    """
    Clase base abstracta (Interfaz) para la carga y estructuración de datos.
    
    Define la firma del método obligatorio que cualquier adaptador de infraestructura
    debe implementar para abastecer a la aplicación con datos preprocesados.
    """

    @abstractmethod
    def cargar(self) -> Dataset:
        """
        Carga los datos desde la fuente configurada, ejecuta el preprocesamiento
        base y devuelve la información encapsulada en una entidad de dominio.

        Returns:
            Dataset: Instancia de la entidad de dominio con las particiones de
                entrenamiento y prueba listas para el modelado.
        """
        pass