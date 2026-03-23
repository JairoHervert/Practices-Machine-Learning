
from __future__ import annotations

from pathlib import Path

import pandas as pd


IRIS_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "species",
]


def load_iris_dataset(path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset Iris descargado externamente desde un archivo CSV.

    Este cargador está pensado para el archivo clásico de UCI
    llamado iris.data, que:
    - no trae encabezados,
    - usa comas como separador,
    - contiene una línea vacía al final.

    Parámetros:
    path : str | Path
        Ruta del archivo descargado.

    Retorna:
    pandas.DataFrame
        DataFrame limpio con nombres de columnas descriptivos.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del dataset: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=IRIS_COLUMNS,
    )

    # El archivo original suele traer una fila vacía al final.
    df = df.dropna(how="all")

    # Eliminar filas incompletas y espacios sobrantes.
    df = df.dropna().copy()
    df["species"] = df["species"].astype(str).str.strip()

    for col in IRIS_COLUMNS[:-1]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target_col: str = "species"):
    """
    Separa un DataFrame en características (X) y etiquetas (y).
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y
