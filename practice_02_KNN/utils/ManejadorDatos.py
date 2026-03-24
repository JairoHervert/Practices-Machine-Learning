import pandas as pd

class ManejadorDatos:
    @staticmethod
    def cargar_datos(ruta_archivo: str):
        """
        Carga el conjunto de datos desde un archivo CSV utilizando pandas,
        separando las características de la variable objetivo. Omite la
        primera columna asumiendo que es un identificador (ID).

        Argumentos:
            ruta_archivo (str): La ruta o nombre del archivo de datos a cargar.

        Retorna:
            tuple: Una tupla que contiene:
                - X (list): Una matriz (lista de listas) con las características del dataset.
                - y (list): Una lista con las etiquetas o categorías objetivo de cada fila.
        """
        try:
            df = pd.read_csv(ruta_archivo, header=None)
            
            X = df.iloc[:, 1:-1].values.tolist()
            y = df.iloc[:, -1].values.tolist()
            
            return X, y
        except FileNotFoundError:
            raise Exception(f"No se encontró el archivo '{ruta_archivo}'. Asegúrate de que esté en la misma carpeta.")
        except Exception as e:
            raise Exception(f"Error al cargar archivo: {e}")