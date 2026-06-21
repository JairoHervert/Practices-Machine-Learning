"""
Módulo de Pruebas: Integración del Pipeline
-------------------------------------------
Este módulo contiene las pruebas de integración que verifican la correcta
interacción entre las distintas capas de la Arquitectura Limpia (Infraestructura,
Dominio y Aplicación) utilizando los componentes reales del sistema.
"""

import os
import pytest
from src.aplicacion.casos_uso.evaluar_comparar import EvaluarYCompararUC
from src.infraestructura.datos.csv_cargador_datos import CsvCargadorDatos
from src.infraestructura.modelos.naive_bayes_clasificador import NaiveBayesClasificador

def test_integracion_pipeline_completo():
    """
    Prueba de integración para el pipeline completo de evaluación y comparación.
    
    Verifica que el sistema pueda cargar el conjunto de datos real desde el disco,
    inicializar un modelo real (Naive Bayes), ejecutar el caso de uso principal
    y devolver métricas calculadas válidas sin lanzar excepciones. 
    
    Nota: Si el archivo de datos no se encuentra en la ruta especificada, 
    la prueba se omite graciosamente (skip) para no romper la suite de pruebas
    en entornos de integración continua (CI/CD) que no posean el CSV.
    """
    ruta_datos = "datos/telco_churn.csv"
    
    # Si no encuentra el dataset real, salta la prueba en lugar de fallar
    if not os.path.exists(ruta_datos):
        pytest.skip("No se encontró el dataset real para la prueba de integración.")

    # 1. Inicializar infraestructura real
    cargador = CsvCargadorDatos(ruta_datos)
    datos = cargador.cargar()

    clasificadores = {
        "NB_Integracion": NaiveBayesClasificador()
    }

    # 2. Ejecutar el caso de uso real
    caso_uso = EvaluarYCompararUC(clasificadores)
    resultados = caso_uso.ejecutar(datos)

    # 3. Verificar que se calcularon métricas válidas
    assert "NB_Integracion" in resultados
    assert resultados["NB_Integracion"].exactitud > 0.0
    assert resultados["NB_Integracion"].auc_roc > 0.0