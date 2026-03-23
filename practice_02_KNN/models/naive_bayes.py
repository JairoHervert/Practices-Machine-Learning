import math
from collections import defaultdict
import numpy as np
import pandas as pd

class NaiveBayes:
    """
    Implementa un clasificador Naive Bayes capaz de trabajar con atributos
    categóricos y continuos dentro del mismo conjunto de datos.

    - Para atributos categóricos usa probabilidades por frecuencia
        con suavizado de Laplace.
    - Para atributos continuos usa una distribución gaussiana
        (media y varianza por clase).

    El modelo detecta automáticamente el tipo de cada atributo
    a partir del DataFrame de entrenamiento.
    """

    def __init__(self, alpha=1.0):
        """
        Inicializa la estructura interna del clasificador.

        Parámetros:
        alpha : float
            Valor del suavizado de Laplace para atributos categóricos.
            Un valor típico es 1.0.

        Atributos principales:
        - target_col: nombre de la columna objetivo.
        - classes_: lista de clases encontradas.
        - class_counts: número de ejemplos por clase.
        - class_priors: probabilidades a priori por clase.
        - feature_types: tipo de cada atributo ('categorical' o 'continuous').
        - categorical_values: valores posibles de cada atributo categórico.
        - categorical_probs: probabilidades condicionales para atributos categóricos.
        - numeric_stats: media y varianza por clase para atributos continuos.
        """
        self.alpha = alpha
        self.target_col = None
        self.classes_ = []
        self.class_counts = {}
        self.class_priors = {}
        self.feature_types = {}
        self.categorical_values = {}
        self.categorical_probs = defaultdict(dict)
        self.numeric_stats = defaultdict(dict)

    def _to_python_scalar(self, value):
        """
        Convierte un escalar de NumPy a un escalar nativo de Python
        cuando sea posible.

        Esto es útil para evitar que en las salidas aparezcan tipos como
        np.int64 o np.float64, y en su lugar se muestren int o float normales.
        """
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def fit(self, df, target_col):
        """
        Entrena el modelo a partir de un DataFrame.

        Parámetros:
        df : pandas.DataFrame
            Conjunto de datos completo, incluyendo atributos y clase.
        target_col : str
            Nombre de la columna objetivo o etiqueta de clase.

        Proceso general:
        1. Separa atributos (X) y clase (y).
        2. Detecta las clases presentes.
        3. Calcula probabilidades a priori P(clase).
        4. Detecta automáticamente si cada atributo es categórico o continuo.
        5. Para atributos categóricos calcula P(valor | clase) con Laplace.
        6. Para atributos continuos calcula media y varianza por clase.

        Retorna:
        self
            El propio objeto entrenado.
        """
        self.target_col = target_col

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        self.classes_ = list(y.unique())
        total = len(df)

        self.class_counts = {c: int((y == c).sum()) for c in self.classes_}
        self.class_priors = {c: self.class_counts[c] / total for c in self.classes_}

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types[col] = "continuous"
            else:
                self.feature_types[col] = "categorical"
                self.categorical_values[col] = sorted(X[col].astype(str).unique().tolist())

        for c in self.classes_:
            Xc = X[y == c]
            nc = len(Xc)

            for col in X.columns:
                if self.feature_types[col] == "categorical":
                    values = self.categorical_values[col]
                    k = len(values)
                    counts = Xc[col].astype(str).value_counts()

                    probs = {}
                    for v in values:
                        count = int(counts.get(v, 0))
                        probs[v] = (count + self.alpha) / (nc + self.alpha * k)

                    self.categorical_probs[c][col] = probs

                else:
                    vals = Xc[col].astype(float)
                    mean = float(vals.mean())
                    var = float(vals.var(ddof=0))

                    if var <= 0 or np.isnan(var):
                        var = 1e-9

                    self.numeric_stats[c][col] = {
                        "mean": mean,
                        "var": var
                    }

        return self

    def _gaussian_pdf(self, x, mean, var):
        """
        Calcula la función de densidad de probabilidad gaussiana
        para un valor continuo.

        Parámetros:
        x : float
            Valor observado.
        mean : float
            Media del atributo para una clase dada.
        var : float
            Varianza del atributo para una clase dada.

        Retorna:
        float
            Valor de la densidad gaussiana en x.

        Esta función se usa para estimar P(x | clase)
        cuando el atributo es continuo.
        """
        coef = 1.0 / math.sqrt(2.0 * math.pi * var)
        exponent = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return coef * exponent


    def predict_one(self, sample, return_details=False):
        """
        Predice la clase de una sola muestra.

        Parámetros:
        sample : dict
            Diccionario con la forma:
            {nombre_atributo: valor, ...}
        return_details : bool
            Si es True, también devuelve detalles del cálculo
            por clase y por atributo.

        Proceso general:
        1. Para cada clase, inicia con log(P(clase)).
        2. Recorre cada atributo de la muestra.
        3. Si el atributo es categórico, usa P(valor | clase).
        4. Si el atributo es continuo, usa la densidad gaussiana.
        5. Suma logaritmos para evitar underflow numérico.
        6. Elige la clase con mayor puntuación final.

        Retorna:
        - Solo la clase predicha, o
        - (clase_predicha, log_scores, details) si return_details=True.
        """
        log_scores = {}
        details = {}

        for c in self.classes_:
            log_score = math.log(self.class_priors[c] + 1e-300)
            detail = {
                "prior": self.class_priors[c],
                "features": []
            }

            for col, value in sample.items():
                if self.feature_types[col] == "categorical":
                    value = str(value)
                    k = len(self.categorical_values[col])
                    nc = self.class_counts[c]

                    prob = self.categorical_probs[c][col].get(
                        value,
                        self.alpha / (nc + self.alpha * k)
                    )

                    log_score += math.log(prob + 1e-300)

                    detail["features"].append({
                        "feature": col,
                        "type": "categorical",
                        "value": value,
                        "prob": prob
                    })

                else:
                    x = float(value)
                    mean = self.numeric_stats[c][col]["mean"]
                    var = self.numeric_stats[c][col]["var"]

                    prob = self._gaussian_pdf(x, mean, var)
                    log_score += math.log(prob + 1e-300)

                    detail["features"].append({
                        "feature": col,
                        "type": "continuous",
                        "value": x,
                        "mean": mean,
                        "var": var,
                        "prob": prob
                    })

            log_scores[c] = log_score
            details[c] = detail

        predicted_class = max(log_scores, key=lambda c: log_scores[c])

        if return_details:
            return predicted_class, log_scores, details
        return predicted_class

    def predict(self, X):
        """
        Predice la clase de varias muestras.

        Parámetros:
        X : pandas.DataFrame o list[dict]
            Puede ser:
            - un DataFrame, donde cada fila es una muestra, o
            - una lista de diccionarios.

        Retorna:
        list
            Lista con las clases predichas para cada muestra.
        """
        if isinstance(X, pd.DataFrame):
            return [self.predict_one(row.to_dict()) for _, row in X.iterrows()]
        elif isinstance(X, list):
            return [self.predict_one(item) for item in X]
        else:
            raise TypeError("X debe ser un DataFrame o una lista de diccionarios")

    def posterior_from_logs(self, log_scores):
        """
        Convierte puntuaciones en logaritmo a probabilidades posteriores
        aproximadas y normalizadas.

        Parámetros:
        log_scores : dict
            Diccionario con la forma:
            {clase: log_score}

        Retorna:
        dict
            Diccionario normalizado:
            {clase: probabilidad_aproximada}

        Se usa principalmente para mostrar resultados de forma más entendible
        al usuario.
        """
        max_log = max(log_scores.values())
        exp_scores = {c: math.exp(v - max_log) for c, v in log_scores.items()}
        total = sum(exp_scores.values())
        return {c: exp_scores[c] / total for c in exp_scores}

    def summary(self):
        """
        Devuelve un resumen simple del modelo entrenado.

        El resumen incluye:
        - clases detectadas
        - probabilidades a priori
        - tipo de cada atributo

        También convierte escalares de NumPy a tipos nativos de Python
        para que la salida sea más limpia.
        """
        classes_py = [self._to_python_scalar(c) for c in self.classes_]
        priors_py = {
            self._to_python_scalar(c): p
            for c, p in self.class_priors.items()
        }

        return {
            "classes": classes_py,
            "class_priors": priors_py,
            "feature_types": self.feature_types
        }