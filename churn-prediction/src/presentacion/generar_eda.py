"""
Módulo de Presentación: Generación de Gráficas del EDA
------------------------------------------------------
Este módulo se encarga de realizar el Análisis Exploratorio de Datos (EDA)
visual, cumpliendo estrictamente con el requerimiento funcional RF-03.
Genera y exporta los artefactos gráficos base para caracterizar la distribución
de la variable objetivo y comprender el impacto de las variables contractuales.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_graficas_eda():
    """
    Genera y guarda en formato físico las gráficas analíticas del EDA.
    
    El proceso lee el conjunto de datos Telco Customer Churn y exporta:
    1. eda_distribucion_fuga.png: Gráfica de barras que muestra el balanceo
       general de la variable objetivo (Churn vs Retención).
    2. eda_fuga_contrato.png: Gráfica cruzada que ilustra la relación directa
       entre la volatilidad de los clientes y sus tipos de contrato.
    """
    ruta = "datos/telco_churn.csv"
    if not os.path.exists(ruta):
        print("Error: No se encontró el dataset.")
        return

    df = pd.read_csv(ruta)
    dir_reportes = "src/presentacion/reportes"
    os.makedirs(dir_reportes, exist_ok=True)

    # Gráfica 1: Distribución general de la fuga
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn Label', palette='Set2')
    plt.title('Distribución de Clientes (Fuga vs Retención)')
    plt.tight_layout()
    plt.savefig(f"{dir_reportes}/eda_distribucion_fuga.png")
    plt.close()

    # Gráfica 2: Fuga según el tipo de contrato
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn Label', palette='Set1')
    plt.title('Fuga de Clientes por Tipo de Contrato')
    plt.tight_layout()
    plt.savefig(f"{dir_reportes}/eda_fuga_contrato.png")
    plt.close()

    print(f"Gráficas de EDA generadas con éxito en '{dir_reportes}'.")

if __name__ == "__main__":
    generar_graficas_eda()