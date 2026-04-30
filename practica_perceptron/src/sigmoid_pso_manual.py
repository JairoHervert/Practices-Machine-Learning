"""
Implementación 3 (Manual): Neurona sigmoidal + PSO sin vectorización de NumPy.

Esta versión evita las bondades de NumPy para el cálculo matricial y utiliza
bucles explícitos (for) para procesar los datos muestra por muestra y 
característica por característica. 

Sirve para demostrar por qué el procesamiento de enjambres suele ser lento
en lenguajes de alto nivel como Python si no se utilizan bibliotecas optimizadas.
"""

import math
import random
import time
from src.metrics import accuracy_score, mean_squared_error

class SigmoidPSOManual:
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
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.c1 = c1
        self.c2 = c2
        self.w_inertial = w_inertial
        self.random_state = random_state
        self.threshold = threshold

        self.weights = None
        self.bias = 0.0
        self.history_ = []
        self.training_time = 0.0

    def sigmoid(self, z: float) -> float:
        """Sigmoide manual."""
        if z < -500: return 0.0
        if z > 500: return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def _compute_output_single(self, sample, weights, bias) -> float:
        """Calcula la salida para una sola muestra usando un bucle manual."""
        z = bias
        for i in range(len(sample)):
            z += sample[i] * weights[i]
        return self.sigmoid(z)

    def _error_function_manual(self, X, y, weights_vector) -> float:
        """
        Calcula el error cuadrático recorriendo toda la matriz X manualmente.
        weights_vector[0] es el bias.
        """
        bias = weights_vector[0]
        weights = weights_vector[1:]
        total_error = 0.0
        
        for i in range(len(X)):
            output = self._compute_output_single(X[i], weights, bias)
            total_error += (y[i] - output) ** 2
            
        return 0.5 * total_error

    def fit(self, X, y):
        start_time = time.time()
        random.seed(self.random_state)
        n_samples = len(X)
        n_features = len(X[0])
        n_dims = n_features + 1

        # Inicialización manual
        particles_pos = []
        particles_vel = []
        for _ in range(self.n_particles):
            pos = [random.uniform(-1.0, 1.0) for _ in range(n_dims)]
            vel = [random.uniform(-0.1, 0.1) for _ in range(n_dims)]
            particles_pos.append(pos)
            particles_vel.append(vel)

        pbest_pos = [p[:] for p in particles_pos]
        pbest_error = [self._error_function_manual(X, y, p) for p in pbest_pos]

        gbest_idx = 0
        for i in range(1, self.n_particles):
            if pbest_error[i] < pbest_error[gbest_idx]:
                gbest_idx = i
        
        gbest_pos = pbest_pos[gbest_idx][:]
        gbest_error_val = pbest_error[gbest_idx]

        self.history_ = []

        for iteration in range(1, self.n_iterations + 1):
            for i in range(self.n_particles):
                # Actualización de velocidad y posición DIMENSIÓN por DIMENSIÓN
                for d in range(n_dims):
                    r1 = random.random()
                    r2 = random.random()
                    
                    cognitive = self.c1 * r1 * (pbest_pos[i][d] - particles_pos[i][d])
                    social = self.c2 * r2 * (gbest_pos[d] - particles_pos[i][d])
                    
                    particles_vel[i][d] = (self.w_inertial * particles_vel[i][d]) + cognitive + social
                    particles_pos[i][d] += particles_vel[i][d]

                # Evaluación manual
                current_error = self._error_function_manual(X, y, particles_pos[i])

                if current_error < pbest_error[i]:
                    pbest_error[i] = current_error
                    pbest_pos[i] = particles_pos[i][:]

                    if current_error < gbest_error_val:
                        gbest_error_val = current_error
                        gbest_pos = particles_pos[i][:]

            # Guardar pesos para predicciones durante el historial
            self.bias = gbest_pos[0]
            self.weights = gbest_pos[1:]
            
            # Nota: Para no ralentizar más el ejemplo, calculamos métricas solo al final
            # o de forma simplificada si fuera necesario, pero mantendremos la estructura.
            self.history_.append({
                "epoch": iteration,
                "accuracy": 0.0, # Se podría calcular pero agregaría más bucles for
                "mse": gbest_error_val / n_samples,
                "error_sum": gbest_error_val
            })

        self.training_time = time.time() - start_time
        return self

    def predict_proba(self, X):
        probs = []
        for i in range(len(X)):
            probs.append(self._compute_output_single(X[i], self.weights, self.bias))
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p >= self.threshold else 0 for p in probs]
