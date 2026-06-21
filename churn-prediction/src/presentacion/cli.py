"""
Módulo de Presentación: Interfaz de Línea de Comandos (CLI)
-----------------------------------------------------------
Este módulo actúa como el punto de entrada principal del sistema.
Orquesta la ejecución del pipeline completo de Machine Learning, 
desde la carga de datos hasta el entrenamiento de modelos, evaluación,
persistencia de modelos entrenados (cumpliendo el requerimiento RF-12) 
y la generación de artefactos gráficos. Además, configura la trazabilidad 
de la ejecución mediante bitácoras o logging (cumpliendo el requerimiento RNF-09).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.metrics import confusion_matrix

from src.infraestructura.datos.csv_cargador_datos import CsvCargadorDatos
from src.infraestructura.modelos.naive_bayes_clasificador import NaiveBayesClasificador
from src.infraestructura.modelos.random_forest_clasificador import RandomForestClasificador
from src.aplicacion.casos_uso.evaluar_comparar import EvaluarYCompararUC

# RNF-09: Configuración de Trazabilidad (Logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Función principal que ejecuta el flujo de trabajo del proyecto.
    
    Realiza los siguientes pasos de forma secuencial:
    1. Carga y preprocesa el dataset Telco Customer Churn.
    2. Inicializa los modelos configurados (Naive Bayes y Random Forest).
    3. Ejecuta el caso de uso central para entrenar y evaluar los modelos.
    4. Imprime las métricas consolidadas en la consola (Exactitud, Precisión, etc.).
    5. Persiste los modelos entrenados en el sistema de archivos mediante joblib.
    6. Genera y guarda las gráficas de las matrices de confusión y el ranking de
       importancia de características en el directorio de reportes.
    """
    logging.info("Iniciando el pipeline de predicción de fuga de clientes...")
    
    ruta_datos = "datos/telco_churn.csv"
    if not os.path.exists(ruta_datos):
        logging.error(f"No se encontró el dataset en '{ruta_datos}'.")
        return

    # Carga de datos
    cargador = CsvCargadorDatos(ruta=ruta_datos)
    datos = cargador.cargar()
    logging.info(f"Datos cargados exitosamente. Muestras de entrenamiento: {datos.X_entrena.shape[0]}")
    
    # Configuración de clasificadores
    rf = RandomForestClasificador()
    nb = NaiveBayesClasificador()
    clasificadores = {"Naive Bayes": nb, "Random Forest": rf}
    
    # Ejecución
    logging.info("Entrenando y evaluando modelos...")
    caso_uso = EvaluarYCompararUC(clasificadores)
    resultados = caso_uso.ejecutar(datos)
    
    print("\n" + "="*40)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("="*40)
    for nombre, metricas in resultados.items():
        print(f"\nModelo: {nombre}")
        print(f"  Exactitud:     {metricas.exactitud:.4f}")
        print(f"  Precisión:     {metricas.precision:.4f}")
        print(f"  Exhaustividad: {metricas.exhaustividad:.4f}")
        print(f"  F1-Score:      {metricas.f1:.4f}")
        print(f"  AUC-ROC:       {metricas.auc_roc:.4f}")
        
    dir_reportes = "src/presentacion/reportes"
    dir_modelos = "src/infraestructura/modelos/persistencia"
    os.makedirs(dir_reportes, exist_ok=True)
    os.makedirs(dir_modelos, exist_ok=True)
    
    # RF-12: Persistencia de modelos entrenados
    joblib.dump(nb._modelo, f"{dir_modelos}/naive_bayes.joblib")
    joblib.dump(rf._modelo, f"{dir_modelos}/random_forest.joblib")
    logging.info(f"Modelos guardados en '{dir_modelos}' (RF-12 cumplido).")

    # Generación de artefactos gráficos
    for nombre, clf in clasificadores.items():
        y_pred = clf.predecir(datos.X_prueba)
        cm = confusion_matrix(datos.y_prueba, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {nombre}')
        plt.tight_layout()
        plt.savefig(f"{dir_reportes}/matriz_confusion_{nombre.replace(' ', '_').lower()}.png")
        plt.close()
        
    importancias = rf.importancia()
    nombres_cols = cargador.nombres_caracteristicas
    indices = np.argsort(importancias)[::-1][:10]
    
    plt.figure(figsize=(10,6))
    plt.title("Top 10 Características más importantes (Random Forest)")
    plt.bar(range(10), importancias[indices], align="center")
    plt.xticks(range(10), [nombres_cols[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{dir_reportes}/importancia_caracteristicas.png")
    plt.close()
    
    logging.info("Artefactos gráficos guardados exitosamente.")
    logging.info("Pipeline completado con éxito.")

if __name__ == "__main__":
    main()