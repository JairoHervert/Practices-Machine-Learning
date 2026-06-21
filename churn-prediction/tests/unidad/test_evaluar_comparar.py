"""
Módulo de Pruebas Unitarias: Evaluación y Comparación
-----------------------------------------------------
Este módulo contiene las pruebas unitarias destinadas a validar la lógica
de negocio de la capa de aplicación (Casos de Uso). Emplea dobles de prueba
(mocks) para aislar el comportamiento del sistema, garantizando que no se
dependa de la infraestructura real ni de la ejecución de algoritmos costosos.
"""

import numpy as np
from unittest.mock import MagicMock
from src.dominio.entidades.dataset import Dataset
from src.aplicacion.casos_uso.evaluar_comparar import EvaluarYCompararUC

def test_ejecutar_evaluacion_con_mocks():
    """
    Verifica la correcta orquestación del caso de uso principal utilizando mocks.
    
    El flujo de la prueba consiste en:
    1. Generar un Dataset simulado con matrices pequeñas de NumPy.
    2. Crear un clasificador falso (MagicMock) que simula predecir con 100% de éxito.
    3. Ejecutar el caso de uso con el clasificador inyectado.
    4. Comprobar que los resultados incluyen al modelo simulado, que la exactitud 
       es perfecta (1.0) y que los métodos de la interfaz abstracta (entrenar y predecir)
       fueron invocados exactamente una vez con los datos correctos.
    """
    # 1. Preparar datos falsos
    X_fake = np.array([[1, 2], [3, 4]])
    y_fake = np.array([0, 1])
    datos_mock = Dataset(X_entrena=X_fake, X_prueba=X_fake, y_entrena=y_fake, y_prueba=y_fake)

    # 2. Preparar clasificador mock (doble de prueba)
    clf_mock = MagicMock()
    clf_mock.predecir.return_value = np.array([0, 1])
    clf_mock.predecir_proba.return_value = np.array([0.1, 0.9])

    clasificadores = {"MockClf": clf_mock}

    # 3. Ejecutar caso de uso
    caso_uso = EvaluarYCompararUC(clasificadores)
    resultados = caso_uso.ejecutar(datos_mock)

    # 4. Verificar que se llamaron a los métodos correctos y las métricas son exactas
    assert "MockClf" in resultados
    assert resultados["MockClf"].exactitud == 1.0
    clf_mock.entrenar.assert_called_once_with(X_fake, y_fake)
    clf_mock.predecir.assert_called_once_with(X_fake)