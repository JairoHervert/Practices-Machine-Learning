"""
Módulo de Presentación: Predicción Individual (Demo interactiva)
----------------------------------------------------------------
Este script demuestra cómo se utilizaría el modelo ya entrenado
(cargando el archivo .joblib) para predecir si un cliente individual
va a cancelar su servicio o no, simulando el entorno de producción.
"""

import joblib
import random
from src.infraestructura.datos.csv_cargador_datos import CsvCargadorDatos

def predecir_cliente_interactivo():
    """
    Simula la llegada de un nuevo cliente al sistema y utiliza el modelo
    Random Forest persistido para emitir una alerta temprana de fuga.
    """
    print("\n" + "="*50)
    print("SISTEMA DE ALERTA TEMPRANA DE FUGA DE CLIENTES")
    print("="*50)
    print("Cargando el modelo Random Forest desde persistencia...")
    
    # 1. Cargar el modelo guardado (.joblib)
    ruta_modelo = "src/infraestructura/modelos/persistencia/random_forest.joblib"
    try:
        modelo_rf = joblib.load(ruta_modelo)
    except FileNotFoundError:
        print("Error: No se encontró el modelo. Asegúrate de ejecutar el CLI principal primero.")
        return

    # 2. Cargar los datos para simular la llegada de un "nuevo cliente"
    # Tomaremos un cliente al azar del conjunto de pruebas (datos no vistos por el modelo).
    cargador = CsvCargadorDatos(ruta="datos/telco_churn.csv")
    dataset = cargador.cargar()
    
    indice_aleatorio = random.randint(0, len(dataset.X_prueba) - 1)
    
    # Extraemos solo a ese cliente (reshape para que scikit-learn lo acepte como 1 sola muestra)
    nuevo_cliente = dataset.X_prueba[indice_aleatorio].reshape(1, -1)
    estado_real = dataset.y_prueba[indice_aleatorio]
    
    print("\n[!] Recibiendo datos de un nuevo cliente desde el CRM...")
    print("Analizando su perfil de facturación, contrato y servicios...\n")
    
    # 3. Hacer la predicción individual
    prediccion = modelo_rf.predict(nuevo_cliente)[0]
    probabilidad = modelo_rf.predict_proba(nuevo_cliente)[0][1] * 100
    
    # 4. Mostrar el resultado de negocio
    print("-" * 50)
    print("RESULTADO DE LA PREDICCIÓN")
    print("-" * 50)
    if prediccion == 1:
        print("[ALERTA] Este cliente tiene ALTO RIESGO de cancelar su servicio.")
        print(f"         Probabilidad de fuga calculada: {probabilidad:.2f}%")
        print("         Acción de Negocio: Enviar promoción de retención inmediatamente.")
    else:
        print("[SEGURO] No hay riesgo inminente de cancelación.")
        print(f"         Probabilidad de fuga calculada: {probabilidad:.2f}%")
        print("         Acción de Negocio: Mantener el servicio estándar.")
        
    print("\n" + "="*50)
    print(f"(Validación para la demo: En la base de datos real, este cliente {'SÍ CANCELÓ' if estado_real == 1 else 'SE QUEDÓ'})")
    print("="*50 + "\n")

if __name__ == "__main__":
    predecir_cliente_interactivo()