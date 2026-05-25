"""
Implementación de Shell Sort instrumentado.

Shell Sort es una mejora del insertion sort que compara elementos separados
por un "gap" o incremento h. Va reduciendo h hasta llegar a 1.

La velocidad del algoritmo depende enormemente de la secuencia de gaps usada.
Encontrar la mejor secuencia es un problema abierto en ciencias de la computación,
y por eso usamos un algoritmo genético para buscarla.

Este módulo expone dos cosas:
    1. shell_sort: ordena un arreglo y devuelve cuántas operaciones realizó.
    2. Secuencias clásicas conocidas (Shell, Knuth, Sedgewick, Ciura) para
       compararlas con la que evolucione nuestro AG.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SortResult:
    """
    Resultado de una corrida de Shell Sort.

    Atributos
    ---------
    comparisons:
        Número de comparaciones realizadas entre elementos.

    swaps:
        Número de movimientos (asignaciones) realizados.

    operations:
        Suma comparisons + swaps. Es la métrica que el AG usará como fitness.

    sorted_array:
        El arreglo ya ordenado (útil para verificar correctness en pruebas).
    """

    comparisons: int
    swaps: int
    operations: int
    sorted_array: list[int]


def shell_sort(array: list[int], gaps: list[int]) -> SortResult:
    """
    Ordena el arreglo usando Shell Sort con la secuencia de gaps proporcionada.

    Parámetros
    ----------
    array:
        Lista de enteros a ordenar. NO se modifica el original (se hace copia).

    gaps:
        Secuencia de incrementos. Debe estar en orden DECRECIENTE y terminar en 1
        para garantizar el ordenamiento correcto.

    Devuelve
    --------
    SortResult con el arreglo ordenado y el conteo de operaciones realizadas.

    Notas
    -----
    Se ignoran los gaps mayores o iguales a len(array), porque comparar elementos
    separados por una distancia mayor al tamaño del arreglo no tiene sentido.
    """

    arr = list(array)   # Copia para no mutar el original
    n = len(arr)

    comparisons = 0
    swaps = 0

    for gap in gaps:
        # Gaps demasiado grandes no aportan trabajo útil
        if gap >= n or gap < 1:
            continue

        # Insertion sort con paso "gap" en lugar de paso 1
        for i in range(gap, n):
            temp = arr[i]
            j = i

            # Vamos retrocediendo de gap en gap mientras el elemento
            # de la izquierda sea mayor que temp
            while j >= gap:
                comparisons += 1
                if arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    swaps += 1
                    j -= gap
                else:
                    break

            if j != i:
                arr[j] = temp
                swaps += 1

    return SortResult(
        comparisons=comparisons,
        swaps=swaps,
        operations=comparisons + swaps,
        sorted_array=arr,
    )


def reversed_array(n: int) -> list[int]:
    """
    Genera un arreglo totalmente desordenado a la inversa.

    Por ejemplo, n=5 devuelve [5, 4, 3, 2, 1]. Este es el peor caso para
    Shell Sort y el que pide el enunciado de la práctica.
    """

    return list(range(n, 0, -1))


# -----------------------------------------------------------------------------
# Secuencias clásicas conocidas
# -----------------------------------------------------------------------------
# Cada función recibe n (tamaño del arreglo) y devuelve la secuencia adaptada.
# Todas terminan en 1 y están en orden decreciente.
# -----------------------------------------------------------------------------

def shell_original_gaps(n: int) -> list[int]:
    """
    Secuencia original de Shell (1959): n/2, n/4, ..., 1.
    Complejidad: O(n^2) en el peor caso. Es la peor de las clásicas.
    """

    gaps = []
    g = n // 2
    while g > 0:
        gaps.append(g)
        g //= 2
    return gaps


def knuth_gaps(n: int) -> list[int]:
    """
    Secuencia de Knuth: h_{k+1} = 3 * h_k + 1, recortada para que h < n.
    Genera: 1, 4, 13, 40, 121, 364, ...
    Complejidad: O(n^1.5).
    """

    gaps = []
    h = 1
    while h < n:
        gaps.append(h)
        h = 3 * h + 1
    return list(reversed(gaps))


def sedgewick_gaps(n: int) -> list[int]:
    """
    Secuencia de Sedgewick (1986). Mezcla dos fórmulas para obtener
    1, 5, 19, 41, 109, 209, 505, ...
    Complejidad: O(n^1.33).
    """

    gaps = []
    k = 0
    while True:
        if k % 2 == 0:
            g = 9 * (2**k - 2**(k // 2)) + 1
        else:
            g = 8 * 2**k - 6 * 2**((k + 1) // 2) + 1
        if g >= n:
            break
        gaps.append(g)
        k += 1
    return list(reversed(gaps))


def ciura_gaps(n: int) -> list[int]:
    """
    Secuencia empírica de Ciura (2001). Encontrada experimentalmente y se
    considera la mejor secuencia conocida para arreglos pequeños y medianos.

    Para n grandes se extiende con factor ~2.25.
    """

    base = [1, 4, 10, 23, 57, 132, 301, 701, 1750]
    extended = list(base)
    while extended[-1] < n:
        extended.append(int(extended[-1] * 2.25))
    return [g for g in reversed(extended) if g < n]
