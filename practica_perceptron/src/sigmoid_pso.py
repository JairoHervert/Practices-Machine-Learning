"""
Implementación 3: Neurona sigmoidal optimizada con Particle Swarm Optimization (PSO).

Este modelo utiliza la misma arquitectura que la neurona sigmoidal:
    z = w1*x1 + w2*x2 + ... + wn*xn + b
    o = 1 / (1 + e^(-z))

Sin embargo, en lugar de usar gradiente descendente, los pesos (incluyendo el
sesgo) se optimizan mediante un algoritmo metaheurístico basado en enjambres.

El PSO mantiene una población de partículas, donde cada partícula representa un
posible vector de pesos. Las partículas se mueven por el espacio de búsqueda
influenciadas por su mejor posición individual (pbest) y la mejor posición
global del enjambre (gbest).

Función objetivo a minimizar (Error):
    E = 1/2 * sum((t - o)^2)

Reglas de actualización:
    v_i(t+1) = z*v_i(t) + c1*r1*(pbest_i - w_i(t)) + c2*r2*(gbest - w_i(t))
    w_i(t+1) = w_i(t) + v_i(t+1)

Donde:
    z: peso inercial
    c1, c2: coeficientes de aceleración (aprendizaje)
    r1, r2: números aleatorios en [0, 1]
"""

import numpy as np

from src.metrics import accuracy_score, mean_squared_error


class SigmoidPSOClassifier:
    """
    Clasificador binario con activación sigmoide y optimización PSO.

    Esta clase sigue la misma interfaz que los modelos anteriores para
    facilitar la comparación.
    """

    def __init__(
        self,
        n_particles: int = 40,
        n_iterations: int = 100,
        c1: float = 1.5,
        c2: float = 1.5,
        w_inertial: float = 0.7,
        random_state: int = 42,
        threshold: float = 0.5,
    ) -> None:
        """
        Inicializa el optimizador PSO.

        Parámetros
        ----------
        n_particles:
            Número de partículas en el enjambre.
        n_iterations:
            Número de iteraciones (generaciones).
        c1:
            Coeficiente cognitivo (atracción hacia el mejor local).
        c2:
            Coeficiente social (atracción hacia el mejor global).
        w_inertial:
            Peso de inercia que controla la exploración vs explotación.
        random_state:
            Semilla para reproducibilidad.
        threshold:
            Umbral para clasificación.
        """

        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w_inertial = w_inertial
        self.random_state = random_state
        self.threshold = threshold

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.history_: list[dict] = []

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Calcula la función sigmoide con protección contra desbordamiento.
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _compute_output(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calcula la salida de la red para un conjunto de pesos dado.
        Asume que el primer elemento de weights es el bias.
        """
        # X tiene n_samples x n_features
        # weights tiene n_features + 1 (incluyendo bias en la posición 0)
        z = X @ weights[1:] + weights[0]
        return self.sigmoid(z)

    def _error_function(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Calcula el error cuadrático tal como se describe en el PDF:
        E = 1/2 * sum((t - o)^2)
        """
        outputs = self._compute_output(X, weights)
        return 0.5 * np.sum((y - outputs) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SigmoidPSOClassifier":
        """
        Entrena el modelo usando PSO para encontrar los pesos óptimos.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        n_dims = n_features + 1  # +1 por el bias

        # 1. Crear una población aleatoria de vectores w
        # Inicializamos posiciones y velocidades
        # Usamos un rango pequeño para los pesos iniciales
        particles_pos = rng.uniform(low=-1.0, high=1.0, size=(self.n_particles, n_dims))
        particles_vel = rng.uniform(low=-0.1, high=0.1, size=(self.n_particles, n_dims))

        # pbest: mejor posición personal de cada partícula
        pbest_pos = particles_pos.copy()
        pbest_error = np.array([self._error_function(X, y, p) for p in pbest_pos])

        # gbest: mejor posición global del enjambre
        gbest_idx = np.argmin(pbest_error)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_error_val = pbest_error[gbest_idx]

        self.history_ = []

        # 5. Repetir g = 100
        for iteration in range(1, self.n_iterations + 1):
            for i in range(self.n_particles):
                # 3. Actualizar población
                # v_i = z*v_i + c1*r1*(pbest_i - w_i) + c2*r2*(gbest - w_i)
                r1 = rng.random(n_dims)
                r2 = rng.random(n_dims)

                cognitive = self.c1 * r1 * (pbest_pos[i] - particles_pos[i])
                social = self.c2 * r2 * (gbest_pos - particles_pos[i])

                particles_vel[i] = (self.w_inertial * particles_vel[i]) + cognitive + social

                # w_i = w_i + v_i
                particles_pos[i] += particles_vel[i]

                # 2. Evaluar cada vector respecto a la función objetivo
                current_error = self._error_function(X, y, particles_pos[i])

                # Actualizar pbest si es mejor
                if current_error < pbest_error[i]:
                    pbest_error[i] = current_error
                    pbest_pos[i] = particles_pos[i].copy()

                    # Actualizar gbest si es mejor que el actual global
                    if current_error < gbest_error_val:
                        gbest_error_val = current_error
                        gbest_pos = particles_pos[i].copy()

            # Registrar historia
            self.weights_ = gbest_pos[1:]
            self.bias_ = gbest_pos[0]
            
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)
            
            self.history_.append({
                "epoch": iteration,
                "accuracy": accuracy_score(y, y_pred),
                "mse": mean_squared_error(y, y_proba),
                "error_sum": gbest_error_val
            })

            # 4. Evaluar si e es suficientemente pequeño (opcional, aquí seguimos hasta g iterations)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las probabilidades (salida sigmoide).
        """
        if self.weights_ is None:
            raise RuntimeError("El modelo todavía no ha sido entrenado.")
        
        z = X @ self.weights_ + self.bias_
        return self.sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna las clases predichas (0 o 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
