# Guía de Ejecución y Análisis - Predicción de Fuga de Clientes

Este documento contiene las instrucciones para ejecutar el pipeline de Machine Learning y una guía rápida para interpretar los artefactos generados.

## 1. Instalación y Preparación

Asegúrate de tener Python instalado y ejecuta los siguientes comandos en tu terminal desde la raíz del proyecto:

1. **Crear un entorno virtual:**
   `python -m venv venv`
   
2. **Activar el entorno virtual:**
   * En Windows: `.\venv\Scripts\activate`
   * En Mac/Linux: `source venv/bin/activate`
   
3. **Instalar las dependencias:**
   `pip install -r requirements.txt`

---

## 2. Comandos de Ejecución

El proyecto está dividido en cuatro flujos principales. Ejecútalos desde la raíz del proyecto utilizando la bandera `-m`:

* **Para generar las gráficas de Análisis Exploratorio (EDA):**
  `python -m src.presentacion.generar_eda`
  *(Esto creará las gráficas base en la carpeta `src/presentacion/reportes`)*

* **Para ejecutar el entrenamiento y evaluación de modelos:**
  `python -m src.presentacion.cli`
  *(Este es el pipeline principal. Verás las métricas en la consola, se guardarán los modelos en `src/infraestructura/modelos/persistencia` y se generarán las matrices y gráficas de importancia en la carpeta de `reportes`)*

* **Para ejecutar la demostración interactiva (Predicción Individual):**
  `python -m src.presentacion.predecir_nuevo_cliente`
  *(Toma un cliente al azar del conjunto de pruebas y evalúa su riesgo de fuga en tiempo real utilizando el modelo previamente entrenado)*

* **Para correr las pruebas unitarias y de integración:**
  `python -m pytest tests/`
  *(Confirma que la lógica de Arquitectura Limpia funciona correctamente)*

---

## 3. Guía de Interpretación de Gráficas (Para el Reporte)

Al correr los scripts, se generarán varias imágenes en la carpeta `src/presentacion/reportes`. Aquí te explico qué significa cada una para que puedas redactar las conclusiones:

### A. Gráficas del EDA (Análisis Exploratorio)
* **`eda_distribucion_fuga.png`:** Muestra cuántos clientes se quedan (0) y cuántos se van (1). 
  * *Para el reporte:* Notarás que hay un desbalance (hay más clientes leales que los que se fugan). Esto es vital mencionarlo porque justifica por qué no solo usamos la métrica de "Exactitud" para evaluar los algoritmos, sino que revisamos el F1-Score y el AUC-ROC.
* **`eda_fuga_contrato.png`:** Compara la fuga de clientes cruzada con su tipo de contrato.
  * *Para el reporte:* Aquí se observa visualmente que los clientes con contratos de "Mes a mes" (Month-to-month) son muchísimo más propensos a cancelar el servicio en comparación con los que tienen contratos de uno o dos años.

### B. Matrices de Confusión (`matriz_confusion_naive_bayes.png` y `matriz_confusion_random_forest.png`)
Esta gráfica es un cuadro dividido en 4 bloques. Representa qué tan bien adivinó el modelo.
* **Cuadro superior izquierdo (Verdaderos Negativos):** Clientes que el modelo dijo que NO se irían, y efectivamente se quedaron.
* **Cuadro inferior derecho (Verdaderos Positivos):** Clientes que el modelo detectó que se irían, y en la vida real sí cancelaron. *(Estos son nuestros aciertos clave)*
* **Cuadro inferior izquierdo (Falsos Negativos):** Clientes que el modelo creyó que se quedarían felices, pero terminaron yéndose. *(El peor error para el negocio, porque no pudimos retenerlos)*
* **Cuadro superior derecho (Falsos Positivos):** Clientes que el modelo marcó en riesgo de irse, pero en realidad no pensaban irse.
* *Para el reporte:* Compara ambos modelos. El modelo que tenga un número más alto en el cuadro inferior derecho y un número más bajo en el inferior izquierdo es el más valioso para la empresa.

### C. Importancia de Características (`importancia_caracteristicas.png`)
Esta gráfica de barras es generada por el Random Forest. Muestra el "Top 10" de las variables que más impactan en la decisión de un cliente para cancelar.
* *Para el reporte:* Las barras más altas son los focos rojos para la empresa de telecomunicaciones. Si variables como `Total Charges` (Cargos Totales) o `Contract` (Tipo de Contrato) están hasta arriba, puedes concluir en el reporte que la estrategia de negocio debe enfocarse en ofrecer descuentos a largo plazo o mejorar las tarifas para retener a los clientes, porque esos son los factores decisivos que los hacen irse.
