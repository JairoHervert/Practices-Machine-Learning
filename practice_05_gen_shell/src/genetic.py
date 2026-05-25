"""
Algoritmo genético para evolucionar secuencias de incrementos de Shell Sort.

Cada individuo de la población es una secuencia de gaps (lista de enteros
estrictamente decreciente que termina en 1). El fitness de un individuo es
el costo (comparaciones + swaps) que Shell Sort gasta al ordenar arreglos
invertidos usando esa secuencia.

El AG minimiza ese costo: entre menos operaciones gaste Shell Sort con la
secuencia, mejor es el individuo.

Flujo general del algoritmo:
    1. Inicializar población aleatoria.
    2. Evaluar fitness de cada individuo.
    3. Repetir por G generaciones:
        a. Conservar a los mejores (elitismo).
        b. Generar el resto vía selección + cruza + mutación.
        c. Evaluar la nueva población.
    4. Devolver al mejor individuo histórico.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .shell_sort import reversed_array, shell_sort


# =============================================================================
# Representación del individuo y fitness
# =============================================================================

# Un individuo es simplemente una lista de enteros. La elegimos así por
# simplicidad: no necesitamos una clase aparte.
Individual = list[int]


def is_valid(individual: Individual) -> bool:
    """
    Un individuo es válido si:
        - Tiene al menos un gen.
        - Termina en 1 (necesario para garantizar el ordenamiento).
        - Está en orden estrictamente decreciente.
        - Todos sus genes son enteros positivos.
    """

    if not individual:
        return False
    if individual[-1] != 1:
        return False
    for a, b in zip(individual, individual[1:]):
        if a <= b:
            return False
    if any(g < 1 for g in individual):
        return False
    return True


def repair(individual: Individual) -> Individual:
    """
    Repara un individuo para que cumpla las restricciones.

    Tras una cruza o mutación, un individuo puede quedar con genes repetidos,
    desordenados o sin terminar en 1. Esta función lo arregla:
        1. Convierte a conjunto (elimina repetidos), añade el 1.
        2. Ordena de forma decreciente.

    Esto evita generar individuos inválidos y mantiene el código simple.
    """

    genes = set(g for g in individual if g >= 1)
    genes.add(1)   # siempre debe terminar en 1
    return sorted(genes, reverse=True)


def fitness(
    individual: Individual,
    test_sizes: list[int],
) -> float:
    """
    Calcula el fitness de un individuo.

    Para cada tamaño en test_sizes, genera el arreglo invertido y ejecuta
    Shell Sort con la secuencia del individuo. Promedia las operaciones
    realizadas. Como queremos MINIMIZAR, un fitness más bajo es mejor.

    Notas
    -----
    Evaluar sobre varios tamaños hace que la secuencia ganadora generalice
    (no se sobreajuste a un solo n).
    """

    total_ops = 0
    for n in test_sizes:
        arr = reversed_array(n)
        result = shell_sort(arr, individual)
        total_ops += result.operations
    return total_ops / len(test_sizes)


# =============================================================================
# Inicialización de la población
# =============================================================================

def random_individual(max_gap: int, rng: random.Random) -> Individual:
    """
    Genera un individuo aleatorio válido.

    Estrategia: elegir entre 3 y 8 gaps aleatorios entre 1 y max_gap,
    siempre incluyendo el 1 y luego reparando.

    El rango 3..8 evita secuencias triviales (de 1 solo gap) o demasiado
    largas (que serían lentas de evaluar).
    """

    n_genes = rng.randint(3, 8)
    genes = {1}
    while len(genes) < n_genes:
        genes.add(rng.randint(2, max_gap))
    return sorted(genes, reverse=True)


def initial_population(
    size: int,
    max_gap: int,
    rng: random.Random,
) -> list[Individual]:
    """Genera una población inicial de individuos aleatorios."""

    return [random_individual(max_gap, rng) for _ in range(size)]


# =============================================================================
# Operadores genéticos: selección, cruza, mutación
# =============================================================================

def tournament_selection(
    population: list[Individual],
    fitnesses: list[float],
    k: int,
    rng: random.Random,
) -> Individual:
    """
    Selección por torneo: elige k individuos al azar y devuelve el de mejor
    fitness (el de menor valor, porque minimizamos).

    Más simple y robusta que la selección por ruleta. El parámetro k controla
    la presión selectiva: k grande = se imponen los fuertes (más explotación),
    k pequeño = más oportunidad para los débiles (más exploración).
    """

    candidates_idx = rng.sample(range(len(population)), k)
    best_idx = min(candidates_idx, key=lambda i: fitnesses[i])
    return list(population[best_idx])   # devolvemos copia


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: random.Random,
) -> Individual:
    """
    Cruza tipo "unión": toma genes de ambos padres y los combina.

    Cada gen del pool conjunto se incluye con probabilidad 0.5. Luego se
    repara para mantener orden y terminar en 1.

    Esta cruza es buena para secuencias de longitud variable porque no
    obliga a partir en un punto fijo (como sí lo haría una cruza de un
    punto, que asume cromosomas de igual longitud).
    """

    pool = set(parent_a) | set(parent_b)
    child = {g for g in pool if rng.random() < 0.5}
    child.add(1)
    return sorted(child, reverse=True)


def mutate(
    individual: Individual,
    max_gap: int,
    mutation_rate: float,
    rng: random.Random,
) -> Individual:
    """
    Aplica una mutación con probabilidad mutation_rate.

    Hay tres tipos de mutación, cada uno con igual probabilidad:
        1. Reemplazar un gen por uno cercano (perturbación pequeña).
        2. Agregar un gen nuevo aleatorio.
        3. Eliminar un gen (excepto el 1).

    Las tres operaciones se reparan al final.
    """

    if rng.random() > mutation_rate:
        return list(individual)   # sin mutación, devolvemos copia

    mutated = list(individual)
    op = rng.choice(["replace", "add", "remove"])

    if op == "replace" and len(mutated) > 1:
        # Reemplaza un gen distinto del 1
        idx = rng.randint(0, len(mutated) - 2)   # no tocar el último (1)
        old = mutated[idx]
        # Perturbación: ±50% del valor original
        delta = max(1, old // 2)
        new_val = max(2, old + rng.randint(-delta, delta))
        mutated[idx] = new_val

    elif op == "add":
        new_gen = rng.randint(2, max_gap)
        mutated.append(new_gen)

    elif op == "remove" and len(mutated) > 2:
        idx = rng.randint(0, len(mutated) - 2)   # no quitar el 1 final
        mutated.pop(idx)

    return repair(mutated)


# =============================================================================
# Bucle principal del algoritmo genético
# =============================================================================

@dataclass
class GAHistory:
    """
    Historial de la evolución del AG, generación por generación.

    Atributos
    ---------
    best_fitness_per_gen:
        Mejor (menor) fitness en cada generación.

    mean_fitness_per_gen:
        Promedio de fitness de la población en cada generación.

    best_individual_per_gen:
        El mejor individuo de cada generación (su secuencia).
    """

    best_fitness_per_gen: list[float] = field(default_factory=list)
    mean_fitness_per_gen: list[float] = field(default_factory=list)
    best_individual_per_gen: list[Individual] = field(default_factory=list)


@dataclass
class GAResult:
    """Resultado final del algoritmo genético."""

    best_individual: Individual
    best_fitness: float
    history: GAHistory


def run_genetic_algorithm(
    population_size: int = 50,
    n_generations: int = 100,
    test_sizes: list[int] | None = None,
    max_gap: int | None = None,
    tournament_k: int = 3,
    mutation_rate: float = 0.3,
    elitism: int = 2,
    random_state: int = 42,
    verbose: bool = True,
) -> GAResult:
    """
    Ejecuta el algoritmo genético completo.

    Parámetros
    ----------
    population_size:
        Cantidad de individuos en cada generación.

    n_generations:
        Cuántas generaciones evolucionar.

    test_sizes:
        Lista de tamaños de arreglo sobre los que se evalúa el fitness.
        Si es None, usa [1000, 5000, 10000].

    max_gap:
        Valor máximo posible para un gap. Si es None, se usa el máximo de
        test_sizes (no tiene sentido un gap mayor al arreglo).

    tournament_k:
        Tamaño del torneo en la selección.

    mutation_rate:
        Probabilidad de que un hijo sufra mutación.

    elitism:
        Número de mejores individuos que pasan directo a la siguiente generación.

    random_state:
        Semilla para reproducibilidad.

    verbose:
        Si True, imprime progreso cada 10 generaciones.

    Devuelve
    --------
    GAResult con el mejor individuo, su fitness y el historial completo.
    """

    if test_sizes is None:
        test_sizes = [1000, 5000, 10000]

    if max_gap is None:
        max_gap = max(test_sizes) // 2

    rng = random.Random(random_state)
    history = GAHistory()

    # Paso 1: población inicial
    population = initial_population(population_size, max_gap, rng)
    fitnesses = [fitness(ind, test_sizes) for ind in population]

    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_overall = list(population[best_idx])
    best_overall_fitness = fitnesses[best_idx]

    if verbose:
        print(f"Generación 0   | mejor: {best_overall_fitness:,.0f} | {best_overall}")

    # Paso 2: bucle de generaciones
    for gen in range(1, n_generations + 1):

        # Ordenamos por fitness (menor es mejor)
        sorted_by_fit = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
        )

        # Elitismo: los mejores pasan directo
        new_population = [list(population[i]) for i in sorted_by_fit[:elitism]]

        # Resto: selección + cruza + mutación
        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, tournament_k, rng)
            parent_b = tournament_selection(population, fitnesses, tournament_k, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, max_gap, mutation_rate, rng)
            new_population.append(child)

        # Evaluamos la nueva generación
        population = new_population
        fitnesses = [fitness(ind, test_sizes) for ind in population]

        # Actualizamos al mejor histórico
        best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[best_idx] < best_overall_fitness:
            best_overall_fitness = fitnesses[best_idx]
            best_overall = list(population[best_idx])

        # Registramos historial
        history.best_fitness_per_gen.append(fitnesses[best_idx])
        history.mean_fitness_per_gen.append(sum(fitnesses) / len(fitnesses))
        history.best_individual_per_gen.append(list(population[best_idx]))

        if verbose and (gen % 10 == 0 or gen == n_generations):
            print(
                f"Generación {gen:3d} | "
                f"mejor: {fitnesses[best_idx]:,.0f} | "
                f"prom: {history.mean_fitness_per_gen[-1]:,.0f} | "
                f"{population[best_idx]}"
            )

    return GAResult(
        best_individual=best_overall,
        best_fitness=best_overall_fitness,
        history=history,
    )
