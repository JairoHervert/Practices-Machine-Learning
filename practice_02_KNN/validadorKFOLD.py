import argparse
import sys
import numpy as np
from sklearn.model_selection import KFold
# Importamos confusion_matrix para poder contar los casos exactos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.ManejadorDatos import ManejadorDatos
from models.KNNPonderado import KNNPonderado

def evaluar_kfold(X, y, k_vecinos=3, n_splits=5):
    """
    Realiza una validación cruzada K-Fold sobre el clasificador KNNPonderado
    y calcula las métricas de rendimiento mostrando porcentajes y fracciones reales.
    """
    X_np = np.array(X)
    y_np = np.array(y)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Listas para los promedios
    exactitudes = []
    precisiones = []
    sensibilidades = []
    puntuaciones_f1 = []
    
    # Contadores para las fracciones totales ("cuántos de cuántos")
    total_aciertos = 0
    total_casos = 0
    total_tp = 0         # Verdaderos Positivos (Predijo M y era M)
    total_pred_m = 0     # Todos los que el modelo dijo que eran M
    total_real_m = 0     # Todos los que REALMENTE eran M en el dataset
    
    print(f"\n--- Iniciando Validación Cruzada {n_splits}-Fold ---")
    print(f"Modelo: K-NN Ponderado (k = {k_vecinos})")
    print("Evaluando, por favor espera (esto puede tomar unos segundos)...\n")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_np), 1):
        
        X_entrenamiento = X_np[train_index].tolist()
        y_entrenamiento = y_np[train_index].tolist()
        
        X_prueba = X_np[test_index].tolist()
        y_prueba = y_np[test_index].tolist()
        
        knn = KNNPonderado(k=k_vecinos)
        knn.entrenar(X_entrenamiento, y_entrenamiento)
        
        y_prediccion = [knn.predecir(x) for x in X_prueba]
        
        # --- CÁLCULO DE FRACCIONES CON MATRIZ DE CONFUSIÓN ---
        # Asumimos 'B' como Negativo y 'M' como Positivo
        cm = confusion_matrix(y_prueba, y_prediccion, labels=['B', 'M'])
        tn, fp, fn, tp = cm.ravel()
        
        aciertos_fold = tp + tn
        casos_fold = len(y_prueba)
        
        total_aciertos += aciertos_fold
        total_casos += casos_fold
        total_tp += tp
        total_pred_m += (tp + fp)
        total_real_m += (tp + fn)
        # -----------------------------------------------------
        
        acc = accuracy_score(y_prueba, y_prediccion)
        prec = precision_score(y_prueba, y_prediccion, pos_label='M', zero_division=0)
        rec = recall_score(y_prueba, y_prediccion, pos_label='M', zero_division=0)
        f1 = f1_score(y_prueba, y_prediccion, pos_label='M', zero_division=0)
        
        exactitudes.append(acc)
        precisiones.append(prec)
        sensibilidades.append(rec)
        puntuaciones_f1.append(f1)
        
        # Mostramos el decimal, el porcentaje y la fracción exacta de este pliegue
        print(f"Pliegue {fold}/{n_splits} -> Exactitud: {acc:.4f} ({acc * 100:.2f}%)  [{aciertos_fold}/{casos_fold} aciertos]")

    # Calculamos promedios
    prom_acc = np.mean(exactitudes)
    std_acc = np.std(exactitudes)
    prom_prec = np.mean(precisiones)
    prom_rec = np.mean(sensibilidades)
    prom_f1 = np.mean(puntuaciones_f1)

    print("\n" + "="*90)
    print("=== RESULTADOS FINALES PROMEDIADOS Y ACUMULADOS ===")
    print("="*90)
    print(f"Exactitud (Accuracy)  : {prom_acc:.4f}  ({prom_acc * 100:.2f}%)  (+/- {std_acc:.4f})  [Total: {total_aciertos}/{total_casos} aciertos]")
    print(f"Precisión (Precision) : {prom_prec:.4f}  ({prom_prec * 100:.2f}%)  [De {total_pred_m} predichos como 'M', acertó en {total_tp}]")
    print(f"Sensibilidad (Recall) : {prom_rec:.4f}  ({prom_rec * 100:.2f}%)  [De {total_real_m} casos reales 'M', logró detectar {total_tp}]")
    print(f"Puntuación F1 (F1)    : {prom_f1:.4f}  ({prom_f1 * 100:.2f}%)  (Equilibrio entre Precisión y Sensibilidad)")
    print("="*90 + "\n")


def principal():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="data/wdbc_modified.data", help="Archivo de datos")
    args = parser.parse_args()

    try:
        X, y = ManejadorDatos.cargar_datos(args.dataset)
        print(f"Dataset '{args.dataset}' cargado. (Total de filas: {len(X)})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        valor_k = input("\nIntroduce el número de vecinos (k) [ej. 5]: ")
        valor_k = int(valor_k) if valor_k.strip() else 3
        
        pliegues = input("Introduce el número de pliegues (Folds) [ej. 5 o 10]: ")
        pliegues = int(pliegues) if pliegues.strip() else 5
    except ValueError:
        print("Valores inválidos. Usando k=3 y Folds=5 por defecto.")
        valor_k = 3
        pliegues = 5

    evaluar_kfold(X, y, k_vecinos=valor_k, n_splits=pliegues)

if __name__ == "__main__":
    principal()