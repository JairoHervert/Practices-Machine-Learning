"""
Implementación 1: Perceptrón con función escalón y regla delta.

El perceptrón es un clasificador lineal binario. Primero calcula una suma
ponderada de las entradas:

    z = w1*x1 + w2*x2 + ... + wn*xn + b

Después aplica una función escalón:

    salida = 1 si z >= 0
    salida = 0 si z < 0

Finalmente, si la predicción no coincide con la etiqueta real, se actualizan
los pesos mediante la regla delta:

    error = t - o

    wi = wi + eta * error * xi
    b  = b  + eta * error

Donde:
    t   = etiqueta real
    o   = salida obtenida por el perceptrón
    eta = tasa de aprendizaje
    xi  = característica i de la muestra
"""

import numpy as np

from src.metrics import accuracy_score, mean_squared_error


class PerceptronDelta:
    """
    Perceptrón binario entrenado con regla delta.

    Esta clase se diseñó para ser sencilla de entender y reutilizable.
    Mantiene una interfaz parecida a modelos comunes de machine learning:

        fit(X, y)      -> entrena el modelo
        predict(X)     -> predice clases 0 o 1
        net_input(X)   -> calcula la suma ponderada
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> None:
        """
        Inicializa el perceptrón.

        Parámetros
        ----------
        learning_rate:
            Tasa de aprendizaje. Controla el tamaño de los ajustes de los pesos.

        n_epochs:
            Número de épocas de entrenamiento.

        random_state:
            Semilla para inicializar pesos de forma reproducible.

        shuffle:
            Si es True, mezcla las muestras en cada época.
        """

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.shuffle = shuffle

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.history_: list[dict] = []

    @staticmethod
    def step_function(z: np.ndarray | float) -> np.ndarray | int:
        """
        Función escalón.

        Si z >= 0, retorna 1.
        Si z < 0, retorna 0.
        """

        if isinstance(z, np.ndarray):
            return (z >= 0).astype(int)

        return int(z >= 0)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la entrada neta del perceptrón:

            z = Xw + b

        Donde:
            X = matriz de características
            w = vector de pesos
            b = sesgo
        """

        if self.weights_ is None:
            raise RuntimeError("El modelo todavía no ha sido entrenado.")

        return X @ self.weights_ + self.bias_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronDelta":
        """
        Entrena el perceptrón usando la regla delta.

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

        # Guardaremos el avance por época para analizar el aprendizaje.
        self.history_ = []

        for epoch in range(1, self.n_epochs + 1):
            errors_in_epoch = 0

            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            for index in indices:
                xi = X[index]
                target = y[index]

                z = float(np.dot(xi, self.weights_) + self.bias_)
                output = self.step_function(z)

                error = target - output

                # Regla delta: ajuste proporcional al error y a la entrada.
                delta = self.learning_rate * error

                self.weights_ += delta * xi
                self.bias_ += delta

                if error != 0:
                    errors_in_epoch += 1

            y_pred = self.predict(X)

            self.history_.append(
                {
                    "epoch": epoch,
                    "errors": errors_in_epoch,
                    "accuracy": accuracy_score(y, y_pred),
                    "mse": mean_squared_error(y, y_pred),
                }
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice la clase de cada muestra.

        Retorna un arreglo con valores 0 o 1.
        """

        z = self.net_input(X)
        return self.step_function(z)