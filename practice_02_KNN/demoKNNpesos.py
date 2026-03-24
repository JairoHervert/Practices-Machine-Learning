import argparse
import sys
from utils.ManejadorDatos import ManejadorDatos
from models.KNNPonderado import KNNPonderado

def principal():
    """
    Punto de entrada principal del programa. 
    Maneja la recepción de argumentos por consola, coordina la carga de datos,
    gestiona la interacción con el usuario para establecer los parámetros, 
    entrena el modelo y muestra los resultados de la predicción.

    Argumentos:
        Ninguno (utiliza sys.argv y la entrada estándar del teclado).

    Retorna:
        None
    """
    print("=== Clasificador K-NN con Pesos (Pandas Mode) ===")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="data/wdbc_modified.data", help="Archivo de datos normalizado")
    args = parser.parse_args()

    try:
        X_entrenamiento, y_entrenamiento = ManejadorDatos.cargar_datos(args.dataset)
        print(f"Dataset '{args.dataset}' cargado correctamente. (Total de filas: {len(X_entrenamiento)})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        valor_k = input("\n1. Introduce el número de vecinos (k) [ej. 5]: ")
        valor_k = int(valor_k) if valor_k.strip() else 3
    except ValueError:
        print("El valor de K debe ser un número entero. Usando k=3 por defecto.")
        valor_k = 3

    knn = KNNPonderado(k=valor_k)
    knn.entrenar(X_entrenamiento, y_entrenamiento)

    print("\n2. Pega la línea de la nueva entrada separada por comas (puedes pegar la fila completa):")
    entrada_cruda = input("> ")
    
    partes = entrada_cruda.replace(" ", "").split(',')

    try:
        if len(partes) == 32:
            nueva_entrada = [float(x) for x in partes[1:-1]]
        elif len(partes) == 30:
            nueva_entrada = [float(x) for x in partes]
        else:
            print(f"\n¡Error!: Se esperaban 30 o 32 valores, pero ingresaste {len(partes)}.")
            return
            
    except ValueError as e:
        print(f"Error al convertir los valores a números: {e}")
        return

    prediccion = knn.predecir(nueva_entrada)

    print("\n" + "="*40)
    print(f"RESULTADO DE LA PREDICCIÓN: {prediccion}")
    
    if str(prediccion) == 'B': 
        print("Clase: BENIGNO")
    elif str(prediccion) == 'M': 
        print("Clase: MALIGNO")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    principal()