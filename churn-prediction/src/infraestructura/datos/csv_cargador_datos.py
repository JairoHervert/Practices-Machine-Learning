"""
Módulo de Infraestructura: Cargador de Datos CSV
------------------------------------------------
Este módulo implementa el adaptador concreto para la carga y preprocesamiento
de datos desde archivos CSV, utilizando pandas y scikit-learn. Representa la
capa externa del sistema encargada de interactuar con el sistema de archivos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.dominio.interfaces.cargador_datos import CargadorDatos
from src.dominio.entidades.dataset import Dataset

class CsvCargadorDatos(CargadorDatos):
    """
    Implementación concreta de CargadorDatos para procesar el dataset Telco Customer Churn.
    
    Esta clase se encarga de leer el archivo CSV, limpiar valores nulos, eliminar
    columnas irrelevantes o que causen fuga de información (data leakage), codificar
    variables categóricas, escalar valores numéricos y particionar los datos de
    manera estratificada y reproducible.
    """

    def __init__(self, ruta: str, semilla: int = 42):
        """
        Inicializa el cargador de datos estableciendo la ruta y la semilla de aleatoriedad.

        Args:
            ruta (str): Ruta relativa o absoluta hacia el archivo CSV de datos.
            semilla (int, opcional): Semilla para fijar la aleatoriedad y garantizar la 
                reproducibilidad en la partición de los datos. Por defecto es 42.
        """
        self.ruta = ruta
        self.semilla = semilla
        self.nombres_caracteristicas = []

    def cargar(self) -> Dataset:
        """
        Ejecuta el pipeline secuencial de carga, limpieza y transformación de datos.

        El proceso abarca 6 etapas:
        1. Limpieza y conversión segura de tipos numéricos.
        2. Eliminación de identificadores y prevención de Data Leakage.
        3. Codificación de variables de texto a numéricas (Label Encoding).
        4. Separación de características (X) y la variable objetivo (y).
        5. Escalamiento estandarizado de las características numéricas (StandardScaler).
        6. Partición estratificada en conjuntos de entrenamiento y prueba.

        Returns:
            Dataset: Entidad de dominio con las matrices (X, y) particionadas y
                listas para el entrenamiento y evaluación.
        """
        # Cargar CSV
        df = pd.read_csv(self.ruta)
        
        # 1. Limpieza y ajuste de tipos
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'] = df['Total Charges'].fillna(0)
        
        # 2. Eliminar identificadores y variables que causan trampa (Data Leakage)
        columnas_a_eliminar = [
            'CustomerID', 'Count', 'Lat Long', 
            'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
        ]
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
        
        # 3. Codificación de variables categóricas (texto a números)
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            
        # 4. Separar características (X) y objetivo (y)
        # En tu dataset, la variable objetivo es 'Churn Value'
        X_df = df.drop('Churn Value', axis=1)
        self.nombres_caracteristicas = X_df.columns.tolist()
        
        X = X_df.values
        y = df['Churn Value'].values
        
        # 5. Escalamiento
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 6. Partición reproducible y estratificada
        X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
            X, y, test_size=0.2, random_state=self.semilla, stratify=y
        )
        
        return Dataset(X_entrena, X_prueba, y_entrena, y_prueba)