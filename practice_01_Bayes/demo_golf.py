import pandas as pd
import unicodedata
from colorama import init, Fore, Style

from Practica_01.naive_bayes import NaiveBayes


# Inicializa Colorama para permitir colores en consola.
# Con autoreset=True, después de cada print el estilo vuelve automáticamente
# al estado normal, evitando que todo el texto posterior conserve el color.

init(autoreset=True)

# Colores
TITLE = Fore.WHITE + Style.BRIGHT
INFO = Fore.BLUE + Style.BRIGHT
OK = Fore.GREEN + Style.BRIGHT
WARN = Fore.YELLOW + Style.BRIGHT
ERR = Fore.RED + Style.BRIGHT
RESET = Style.RESET_ALL


def normalize_text(text):
    """
    Normaliza una cadena de texto para facilitar comparaciones.

    Proceso que realiza:
    1. Convierte el valor a cadena.
    2. Elimina espacios al inicio y al final.
    3. Convierte todo a minúsculas.
    4. Elimina acentos y diacríticos usando normalización Unicode.

    Esto permite que entradas como:
    ' Frío ', 'frio' y 'FRÍO'
    se traten como el mismo valor lógico.

    Parámetros:
    text : any
        Valor que se desea normalizar.

    Retorna:
    str
        Cadena normalizada.
    """
    text = str(text).strip().lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text


def print_separator():
    print(TITLE + "=" * 70)


def load_golf_dataset(path="dataset_golf.csv"):
    """
    Carga y prepara el dataset de golf desde un archivo CSV.

    Tareas que realiza:
    1. Lee el archivo con pandas.
    2. Normaliza los nombres de columnas.
    3. Normaliza los textos de las columnas categóricas.
    4. Elimina la columna 'dia' si existe, ya que normalmente es
        un identificador y no una característica útil para clasificar.

    Parámetros:
    path : str
        Ruta del archivo CSV a cargar.

    Retorna:
    pandas.DataFrame
        DataFrame limpio y listo para entrenar el modelo.
    """
    df = pd.read_csv(path)

    # Normalizar nombres de columnas
    df.columns = [normalize_text(col) for col in df.columns]

    # Normalizar texto en columnas tipo object
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).apply(normalize_text)

    # Eliminar columna dia si existe
    if "dia" in df.columns:
        df = df.drop(columns=["dia"])

    return df


def ask_categorical_input(feature_name, valid_options):
    """
    Solicita al usuario un valor categórico desde teclado.

    La función:
    1. Muestra el nombre de la característica y sus valores conocidos.
    2. Normaliza la entrada del usuario.
    3. Verifica si el valor existe en el entrenamiento.
    4. Si no existe, advierte al usuario y permite continuar
        usando suavizado de Laplace.

    Parámetros:
    feature_name : str
        Nombre de la característica que se va a capturar.
    valid_options : list[str]
        Lista de valores observados en el entrenamiento para esa característica.

    Retorna:
    str
        Valor ingresado por el usuario, ya normalizado.
    """
    options_text = "/".join(valid_options)
    while True:
        value = input(INFO + f"{feature_name} ({options_text}): " + RESET)
        value = normalize_text(value)

        # Permitimos valores no vistos, pero advertimos
        if value not in valid_options:
            print(WARN + f"Aviso: '{value}' no aparece en el entrenamiento para '{feature_name}'.")
            print(WARN + "Se usará suavizado de Laplace si decides continuar.")
            confirm = input(INFO + "¿Deseas usarlo de todos modos? (s/n): " + RESET).strip().lower()
            if confirm == "s":
                return value
        else:
            return value


def show_result(model, pred, log_scores, details):
    """
    Muestra en consola el resultado de la clasificación.

    La función imprime:
    1. La clase predicha.
    2. Las probabilidades posteriores aproximadas por clase.
    3. El detalle de cálculo para cada clase y cada característica.

    Parámetros:
    model : NaiveBayes
        Modelo ya entrenado.
    pred : str
        Clase predicha por el modelo.
    log_scores : dict
        Puntajes en logaritmo para cada clase.
    details : dict
        Información detallada del cálculo por clase, generada
        por predict_one(..., return_details=True).
    """
    posterior = model.posterior_from_logs(log_scores)

    print_separator()
    print(TITLE + "RESULTADO FINAL")
    print_separator()

    if pred == "si":
        print(OK + f"Clase predicha: {pred.upper()}")
    else:
        print(ERR + f"Clase predicha: {pred.upper()}")

    print(INFO + "\nProbabilidades posteriores aproximadas:")
    for c, p in posterior.items():
        color = OK if c == pred else WARN
        print(color + f"  {c}: {p * 100:.2f}%")

    print_separator()
    print(TITLE + "DETALLE POR CLASE")
    print_separator()

    for c, info in details.items():
        color = OK if c == pred else WARN
        print(color + f"\nClase: {c}")
        print(color + f"P({c}) = {info['prior']:.6f}")

        for feat in info["features"]:
            print(
                f"  - {feat['feature']} = {feat['value']}  "
                f"=> P({feat['feature']}={feat['value']} | {c}) = {feat['prob']:.6f}"
            )


def main():
    """
    Función principal del programa interactivo para el dataset de golf.

    Flujo general:
    1. Carga y limpia el dataset.
    2. Entrena el modelo Naive Bayes.
    3. Muestra un resumen del modelo entrenado.
    4. Permite al usuario capturar una nueva entrada desde teclado.
    5. Realiza la predicción y muestra el resultado.
    6. Repite el proceso hasta que el usuario decida salir.
    """
    df = load_golf_dataset("dataset_golf.csv")

    model = NaiveBayes(alpha=1.0)
    model.fit(df, target_col="juego")

    print_separator()
    print(TITLE + "NAIVE BAYES - DATASET GOLF")
    print_separator()

    print(INFO + "Resumen del modelo:")
    print(model.summary())

    print(INFO + "\nValores detectados:")
    for col, values in model.categorical_values.items():
        print(f"  {col}: {values}")

    while True:
        print_separator()
        print(TITLE + "CAPTURA DE NUEVA ENTRADA")
        print_separator()

        sample = {}
        for feature, ftype in model.feature_types.items():
            if ftype == "categorical":
                sample[feature] = ask_categorical_input(
                    feature,
                    model.categorical_values[feature]
                )

        pred, log_scores, details = model.predict_one(sample, return_details=True)
        show_result(model, pred, log_scores, details)

        again = input(INFO + "\n¿Deseas probar otra entrada? (s/n): " + RESET).strip().lower()
        if again != "s":
            print(OK + "\nPrograma terminado.")
            break


if __name__ == "__main__":
    main()

