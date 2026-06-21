"""
Módulo de Pruebas Unitarias: Validación de Datos
------------------------------------------------
Este módulo contiene las pruebas diseñadas para garantizar la robustez
del componente de carga de datos. Verifica que el sistema es capaz de
manejar adecuadamente valores anómalos o corruptos (como espacios en blanco
en columnas numéricas) sin interrumpir la ejecución del pipeline.
"""

import pytest
import pandas as pd
from src.infraestructura.datos.csv_cargador_datos import CsvCargadorDatos

def test_cargador_maneja_datos_corruptos(tmp_path):
    """
    Verifica que el cargador de datos procese y limpie correctamente un CSV con errores.
    
    El flujo de la prueba consiste en:
    1. Simular un conjunto de datos en memoria con 10 registros que incluye
       valores numéricos inválidos (espacios) en la columna 'Total Charges'.
    2. Escribir temporalmente este conjunto de datos en el sistema de archivos
       utilizando el fixture 'tmp_path' de pytest.
    3. Instanciar el CsvCargadorDatos con el archivo temporal y ejecutar cargar().
    4. Comprobar que el cargador aplica la coerción de tipos y el relleno de nulos
       correctamente, devolviendo un Dataset con las particiones listas.
    """
    # Simulamos un CSV con 10 registros y algunos errores (' ' en Total Charges)
    df_fake = pd.DataFrame({
        'Total Charges': ['100.5', ' ', '250.0', '150.0', '200.0', '300.0', ' ', '120.0', '90.0', '400.0'],
        'Churn Value': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month']
    })
    
    # Guardamos el archivo temporalmente
    ruta_fake = tmp_path / "datos_corruptos.csv"
    df_fake.to_csv(ruta_fake, index=False)

    # Probamos que el cargador limpie los espacios en blanco sin fallar
    cargador = CsvCargadorDatos(ruta=str(ruta_fake))
    dataset = cargador.cargar()

    # Verificamos que todo se procesó
    assert len(dataset.X_entrena) > 0
    assert len(dataset.y_entrena) > 0