import math
from collections import defaultdict

class KNNPonderado:
    def __init__(self, k: int = 3):
        """
        Inicializa el clasificador K-Nearest Neighbors (K-NN) con cálculo de pesos.

        Argumentos:
            k (int, opcional): El número de vecinos más cercanos a considerar 
                               para la clasificación. Por defecto es 3.
                               
        Retorna:
            None
        """
        self.k = k
        self.X_entrenamiento = []
        self.y_entrenamiento = []

    def entrenar(self, X, y):
        """
        Almacena los datos de entrenamiento y sus respectivas etiquetas.
        En el algoritmo K-NN original, esta fase solo consiste en memorizar los datos.

        Argumentos:
            X (list): Matriz de características de los datos de entrenamiento.
            y (list): Lista de etiquetas objetivo correspondientes a 'X'.

        Retorna:
            None
        """
        self.X_entrenamiento = X
        self.y_entrenamiento = y

    def _distancia_euclidiana(self, p1, p2):
        """
        Calcula la distancia euclidiana entre dos puntos (vectores) en un espacio n-dimensional.

        Argumentos:
            p1 (list o tuple): Las coordenadas del primer punto.
            p2 (list o tuple): Las coordenadas del segundo punto.

        Retorna:
            float: El valor de la distancia euclidiana entre ambos puntos.
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def predecir(self, x_nuevo):
        """
        Predice la clase de una nueva entrada evaluando las distancias a sus 'k' vecinos 
        más cercanos. Asigna un peso a cada vecino (mayor peso a menor distancia) para 
        desempatar y afinar la clasificación.

        Argumentos:
            x_nuevo (list): El vector de características del nuevo dato a clasificar.

        Retorna:
            str/Any: La clase o categoría predicha con el mayor peso acumulado.
        """
        distancias = []
        for i, x_t in enumerate(self.X_entrenamiento):
            dist = self._distancia_euclidiana(x_nuevo, x_t)
            distancias.append((dist, self.y_entrenamiento[i]))
        
        distancias.sort(key=lambda x: x[0])
        k_cercanos = distancias[:self.k]
        
        pesos_clases = defaultdict(float)
        d1 = k_cercanos[0][0]  
        dk = k_cercanos[-1][0] 

        for dist, etiqueta in k_cercanos:
            if dk != d1:
                peso = (dk - dist) / (dk - d1)
            else:
                peso = 1.0 
            pesos_clases[etiqueta] += peso
            
        return max(pesos_clases.items(), key=lambda x: x[1])[0]