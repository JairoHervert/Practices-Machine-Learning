"""
Archivo principal de ejecución de la práctica 5.

Algoritmo genético para encontrar una buena secuencia de incrementos para
Shell Sort, aplicado al ordenamiento de arreglos totalmente desordenados a
la inversa.

Flujo general:
    1. Configurar el AG (población, generaciones, tamaños de prueba).
    2. Ejecutar la evolución.
    3. Comparar la secuencia evolucionada contra secuencias clásicas
       (Shell original, Knuth, Sedgewick, Ciura).
    4. Graficar la curva de evolución del fitness.
    5. Graficar la comparación con secuencias clásicas en un n grande.

Nota sobre el tamaño n del enunciado
------------------------------------
El enunciado pide n = 10^20, pero ese tamaño es físicamente imposible de
instanciar (requeriría ~800 exabytes de memoria y siglos de cómputo). En su
lugar evaluamos sobre tamaños tratables (1000, 5000, 10000) durante la
evolución, y validamos con un n grande (100000) al final. La secuencia
ganadora generaliza porque la fórmula de crecimiento es independiente del
tamaño concreto del arreglo.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.genetic import run_genetic_algorithm
from src.shell_sort import (
    ciura_gaps,
    knuth_gaps,
    reversed_array,
    sedgewick_gaps,
    shell_original_gaps,
    shell_sort,
)


# Valores por defecto del experimento
# -----------------------------------
# Estos defaults están elegidos para que la práctica corra en ~1-3 minutos.
# Todos son sobreescribibles desde la línea de comandos. Ver --help.
DEFAULT_POPULATION_SIZE = 30
DEFAULT_N_GENERATIONS = 40
DEFAULT_TEST_SIZES = [500, 2000, 10000]
DEFAULT_VALIDATION_N = 10_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_MUTATION_RATE = 0.3
DEFAULT_TOURNAMENT_K = 3
DEFAULT_ELITISM = 2

def evaluate_classical_sequences(n: int) -> dict[str, dict]:
    """
    Evalúa todas las secuencias clásicas sobre un arreglo invertido de tamaño n.

    Devuelve un diccionario con el nombre de cada secuencia, sus gaps,
    el número de operaciones que gastó y una verificación de que el
    arreglo quedó realmente ordenado.
    """

    arr = reversed_array(n)
    expected = sorted(arr)

    sequences = {
        "Shell original": shell_original_gaps(n),
        "Knuth": knuth_gaps(n),
        "Sedgewick": sedgewick_gaps(n),
        "Ciura": ciura_gaps(n),
    }

    results = {}
    for name, gaps in sequences.items():
        result = shell_sort(arr, gaps)
        results[name] = {
            "gaps": gaps,
            "operations": result.operations,
            "comparisons": result.comparisons,
            "swaps": result.swaps,
            "correct": result.sorted_array == expected,
        }
    return results


def plot_evolution(history, save_path: Path) -> None:
    """
    Grafica la evolución del fitness: mejor y promedio por generación.

    La curva del 'mejor' debe ir bajando (o quedarse igual gracias al elitismo).
    La curva del 'promedio' indica si toda la población está mejorando o solo
    el líder.
    """

    generations = list(range(1, len(history.best_fitness_per_gen) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(
        generations,
        history.best_fitness_per_gen,
        label="Mejor de la generación",
        linewidth=2,
    )
    plt.plot(
        generations,
        history.mean_fitness_per_gen,
        label="Promedio de la población",
        linewidth=2,
        alpha=0.6,
    )
    plt.xlabel("Generación")
    plt.ylabel("Fitness (operaciones, menor es mejor)")
    plt.title("Evolución del algoritmo genético")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_comparison(
    classical: dict[str, dict],
    ga_name: str,
    ga_ops: int,
    n: int,
    save_path: Path,
) -> None:
    """
    Gráfica de barras comparando operaciones de cada secuencia.
    """

    names = list(classical.keys()) + [ga_name]
    ops = [classical[k]["operations"] for k in classical] + [ga_ops]

    colors = ["#888888"] * len(classical) + ["#2E8B57"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, ops, color=colors)
    plt.ylabel("Operaciones (comparaciones + swaps)")
    plt.title(f"Comparación de secuencias en n = {n:,}")
    plt.xticks(rotation=15)

    # Etiqueta encima de cada barra
    for bar, value in zip(bars, ops):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def print_section(title: str) -> None:
    """Imprime un encabezado bonito de sección."""

    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos.

    Ejemplos de uso:
        python main.py
        python main.py --population 50 --generations 100
        python main.py --test-sizes 1000 5000 10000 --validation-n 50000
        python main.py --mutation-rate 0.5 --seed 123 --quiet
    """

    parser = argparse.ArgumentParser(
        description="Algoritmo Genético para encontrar secuencias de Shell Sort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Parámetros del AG ---
    parser.add_argument(
        "--population", "-p",
        type=int, default=DEFAULT_POPULATION_SIZE,
        help="Tamaño de la población.",
    )
    parser.add_argument(
        "--generations", "-g",
        type=int, default=DEFAULT_N_GENERATIONS,
        help="Número de generaciones.",
    )
    parser.add_argument(
        "--mutation-rate", "-m",
        type=float, default=DEFAULT_MUTATION_RATE,
        help="Probabilidad de mutación (entre 0 y 1).",
    )
    parser.add_argument(
        "--tournament-k", "-k",
        type=int, default=DEFAULT_TOURNAMENT_K,
        help="Tamaño del torneo en la selección.",
    )
    parser.add_argument(
        "--elitism", "-e",
        type=int, default=DEFAULT_ELITISM,
        help="Número de mejores individuos que pasan directo a la siguiente generación.",
    )

    # --- Tamaños de arreglo ---
    parser.add_argument(
        "--test-sizes", "-t",
        type=int, nargs="+", default=DEFAULT_TEST_SIZES,
        help="Tamaños de arreglo usados durante el entrenamiento. Acepta varios valores: -t 500 2000 10000",
    )
    parser.add_argument(
        "--validation-n", "-v",
        type=int, default=DEFAULT_VALIDATION_N,
        help="Tamaño del arreglo usado solo para la comparación final.",
    )

    # --- Reproducibilidad y salida ---
    parser.add_argument(
        "--seed", "-s",
        type=int, default=DEFAULT_RANDOM_STATE,
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Silenciar el progreso por generación del AG.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str, default="plots",
        help="Carpeta donde guardar las gráficas.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)

    print_section("PRÁCTICA 5 — Algoritmo Genético para Shell Sort")
    print(f"Población:         {args.population}")
    print(f"Generaciones:      {args.generations}")
    print(f"Tasa de mutación:  {args.mutation_rate}")
    print(f"Tamaño de torneo:  {args.tournament_k}")
    print(f"Elitismo:          {args.elitism}")
    print(f"Tamaños prueba:    {args.test_sizes}")
    print(f"Tamaño validación: {args.validation_n:,}")
    print(f"Semilla:           {args.seed}")

    # -------------------------------------------------------------------------
    # 1. Ejecutar el algoritmo genético
    # -------------------------------------------------------------------------
    print_section("Evolución del algoritmo genético")

    result = run_genetic_algorithm(
        population_size=args.population,
        n_generations=args.generations,
        test_sizes=args.test_sizes,
        tournament_k=args.tournament_k,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        random_state=args.seed,
        verbose=not args.quiet,
    )

    print_section("Mejor individuo encontrado")
    print(f"Secuencia: {result.best_individual}")
    print(f"Longitud:  {len(result.best_individual)} gaps")
    print(f"Fitness (promedio de ops sobre {args.test_sizes}): {result.best_fitness:,.0f}")

    # -------------------------------------------------------------------------
    # 2. Comparar contra secuencias clásicas sobre n grande
    # -------------------------------------------------------------------------
    print_section(f"Comparación con secuencias clásicas en n = {args.validation_n:,}")

    classical = evaluate_classical_sequences(args.validation_n)

    # Ejecutar nuestra secuencia sobre el mismo n
    arr = reversed_array(args.validation_n)
    expected = sorted(arr)
    ga_result = shell_sort(arr, result.best_individual)
    ga_correct = ga_result.sorted_array == expected

    print(f"\n{'Secuencia':<20} {'Operaciones':>15} {'Correcto':>10}")
    print("-" * 50)
    for name, data in classical.items():
        ok = "Sí" if data["correct"] else "NO"
        print(f"{name:<20} {data['operations']:>15,} {ok:>10}")

    ok = "Sí" if ga_correct else "NO"
    print(f"{'AG (evolucionado)':<20} {ga_result.operations:>15,} {ok:>10}")

    # -------------------------------------------------------------------------
    # 3. Gráficas
    # -------------------------------------------------------------------------
    print_section("Generando gráficas")

    evolution_path = plots_dir / "evolucion_ag.png"
    comparison_path = plots_dir / "comparacion_secuencias.png"

    plot_evolution(result.history, evolution_path)
    print(f"Gráfica de evolución guardada en: {evolution_path}")

    plot_comparison(
        classical=classical,
        ga_name="AG",
        ga_ops=ga_result.operations,
        n=args.validation_n,
        save_path=comparison_path,
    )
    print(f"Gráfica de comparación guardada en: {comparison_path}")

    print_section("Práctica completada")


if __name__ == "__main__":
    main()
