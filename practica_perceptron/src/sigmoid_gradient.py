"""
Implementación 2: Neurona sigmoidal con gradiente descendente.

Este modelo es similar al perceptrón en el sentido de que calcula una suma
ponderada de las entradas:

    z = w1*x1 + w2*x2 + ... + wn*xn + b

La diferencia principal es que aquí no se usa una función escalón, sino una
función sigmoide:

    sigmoid(z) = 1 / (1 + e^(-z))

La sigmoide produce valores continuos entre 0 y 1, por lo que su salida puede
interpretarse como una probabilidad aproximada de pertenecer a la clase 1.

Para entrenar el modelo se usa error cuadrático medio. Para una muestra:

    E = 1/2 * (t - o)^2

Donde:
    t = etiqueta real
    o = salida de la sigmoide

Como la sigmoide es derivable, se puede usar gradiente descendente para ajustar
los pesos. La derivada de la sigmoide es:

    sigmoid'(z) = o * (1 - o)

Por lo tanto, la actualización usada es:

    delta = (t - o) * o * (1 - o)

    wi = wi + eta * delta * xi
    b  = b  + eta * delta

En esta implementación se usa gradiente estocástico (SGD): los pesos se
actualizan muestra por muestra, no con el promedio de todo el lote. Esto
corresponde a lo que el profesor llamó "gradiente estocástico":

    E_d(w) = 1/2 * (t_d - o_d)^2   <- error de UNA muestra d

El modelo procesa una muestra, actualiza pesos, luego pasa a la siguiente.
"""

import numpy as np

from src.metrics import accuracy_score, mean_squared_error


class SigmoidGradientClassifier:
    """
    Clasificador binario con activación sigmoide y gradiente descendente.

    Esta clase mantiene una interfaz parecida al perceptrón:

        fit(X, y)           -> entrena el modelo
        predict(X)          -> predice clases 0 o 1
        predict_proba(X)    -> devuelve salidas continuas entre 0 y 1
        net_input(X)        -> calcula la suma ponderada

    La salida continua se convierte a clase usando un umbral, por defecto 0.5.
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        n_epochs: int = 1500,
        random_state: int = 42,
        threshold: float = 0.5,
    ) -> None:
        """
        Inicializa el clasificador sigmoidal.

        Parámetros
        ----------
        learning_rate:
            Tasa de aprendizaje. Controla el tamaño del ajuste de los pesos.

        n_epochs:
            Número de épocas de entrenamiento.

        random_state:
            Semilla para inicializar pesos de forma reproducible.

        threshold:
            Umbral para convertir la salida sigmoide en clase.
            Si salida >= threshold, se predice 1.
            Si salida < threshold, se predice 0.
        """

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.threshold = threshold

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.history_: list[dict] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Calcula la función sigmoide.

        Se usa np.clip para limitar valores demasiado grandes o pequeños de z.
        Esto evita problemas numéricos con la función exponencial.
        """

        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la entrada neta del modelo:

            z = Xw + b

        Donde:
            X = matriz de características
            w = vector de pesos
            b = sesgo
        """

        if self.weights_ is None:
            raise RuntimeError("El modelo todavía no ha sido entrenado.")

        return X @ self.weights_ + self.bias_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SigmoidGradientClassifier":
        """
        Entrena la neurona sigmoidal mediante gradiente descendente.

        Parámetros
        ----------
        X:
            Matriz de entrenamiento.

        y:
            Etiquetas reales del conjunto de entrenamiento.

        Retorna
        -------
        self:
            El propio modelo entrenado.
        """

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # Inicializamos pesos pequeños de forma aleatoria.
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0

        # Convertimos y a float para operar con salidas continuas.
        y_float = y.astype(float)

        self.history_ = []

        for epoch in range(1, self.n_epochs + 1):

            # Mezclamos el orden de las muestras en cada época.
            # Esto evita que el modelo aprenda el orden de los datos.
            indices = rng.permutation(n_samples)

            # --- Gradiente estocástico: una muestra a la vez ---
            # A diferencia del batch, aquí los pesos cambian después
            # de CADA muestra, no al final de la época completa.
            for i in indices:
                xi = X[i]           # una muestra: vector de 30 características
                ti = y_float[i]     # su etiqueta real (0 o 1)

                # Paso 1: calcular la salida de la neurona para esta muestra
                zi = float(np.dot(xi, self.weights_) + self.bias_)
                oi = float(self.sigmoid(zi))

                # Paso 2: calcular delta para esta muestra
                # delta = (t - o) * o * (1 - o)
                #   (t - o)    -> qué tan equivocada estuvo la predicción
                #   o * (1-o)  -> derivada de la sigmoide (sigma')
                delta_i = (ti - oi) * oi * (1.0 - oi)

                # Paso 3: actualizar pesos con ESTA muestra
                # delta_wi = eta * delta * xi
                self.weights_ += self.learning_rate * delta_i * xi
                self.bias_    += self.learning_rate * delta_i

            # Al final de la época registramos métricas globales
            probabilities = self.predict_proba(X)
            predictions = self.predict(X)

            self.history_.append(
                {
                    "epoch": epoch,
                    "accuracy": accuracy_score(y, predictions),
                    "mse": mean_squared_error(y_float, probabilities),
                }
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve la salida continua del modelo.

        Cada valor está entre 0 y 1.
        """

        z = self.net_input(X)
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Convierte la salida sigmoide en una clase 0 o 1.

        Si la probabilidad es mayor o igual al umbral, se predice 1.
        En caso contrario, se predice 0.
        """

        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)