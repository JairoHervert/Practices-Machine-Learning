from colorama import init, Fore, Style
from sklearn.datasets import load_iris

from Practica_01.naive_bayes import NaiveBayes

init(autoreset=True)


# Colores
TITLE = Fore.WHITE + Style.BRIGHT
INFO = Fore.BLUE + Style.BRIGHT
OK = Fore.GREEN + Style.BRIGHT
WARN = Fore.YELLOW + Style.BRIGHT
ERR = Fore.RED + Style.BRIGHT
RESET = Style.RESET_ALL


def print_separator():
    print(TITLE + "=" * 70)


def ask_float(feature_name):
    """
    Solicita al usuario un valor numérico de tipo float.

    La función se usa para capturar las características continuas
    del dataset Iris:
    - sepal length (cm)
    - sepal width (cm)
    - petal length (cm)
    - petal width (cm)

    Si el usuario escribe un valor inválido, se muestra un mensaje
    de error y se vuelve a pedir el dato.

    Parámetros:
    feature_name : str
        Nombre de la característica que se desea capturar.

    Retorna:
    float
        Valor numérico ingresado por el usuario.
    """
    while True:
        try:
            value = float(input(INFO + f"{feature_name}: " + RESET))
            return value
        except ValueError:
            print(ERR + "Entrada inválida. Debes escribir un número.")


def show_result(model, pred, log_scores, details, target_names):
    """
    Muestra en consola el resultado de la clasificación de una flor.

    La función imprime:
    1. La clase predicha en formato numérico y textual.
    2. Las probabilidades posteriores aproximadas por clase.
    3. El detalle del cálculo para cada clase y cada característica.

    Parámetros:
    model : NaiveBayes
        Modelo Naive Bayes ya entrenado.
    pred : int
        Clase predicha por el modelo.
    log_scores : dict
        Puntajes en logaritmo obtenidos para cada clase.
    details : dict
        Detalle del cálculo por clase y por atributo.
    target_names : array-like
        Nombres reales de las clases del dataset Iris.
    """
    posterior = model.posterior_from_logs(log_scores)

    pred_name = target_names[int(pred)]

    print_separator()
    print(TITLE + "RESULTADO FINAL")
    print_separator()

    print(OK + f"Clase predicha: {int(pred)} -> {pred_name}")

    print(INFO + "\nProbabilidades posteriores aproximadas:")
    for c, p in posterior.items():
        class_name = target_names[int(c)]
        color = OK if c == pred else WARN
        print(color + f"  {int(c)} -> {class_name}: {p * 100:.2f}%")

    print_separator()
    print(TITLE + "DETALLE POR CLASE")
    print_separator()

    for c, info in details.items():
        class_name = target_names[int(c)]
        color = OK if c == pred else WARN

        print(color + f"\nClase: {int(c)} -> {class_name}")
        print(color + f"P(clase) = {info['prior']:.6f}")

        for feat in info["features"]:
            print(
                f"  - {feat['feature']}: x={feat['value']:.4f}, "
                f"media={feat['mean']:.4f}, var={feat['var']:.4f}, "
                f"pdf={feat['prob']:.10f}"
            )


def main():
    """
    Función principal del programa interactivo para el dataset Iris.

    Flujo general:
    1. Carga el dataset Iris como DataFrame.
    2. Obtiene los nombres de las clases.
    3. Entrena el modelo Naive Bayes con la columna objetivo 'target'.
    4. Muestra un resumen del modelo entrenado.
    5. Solicita al usuario una nueva flor desde teclado.
    6. Realiza la predicción y muestra el resultado.
    7. Repite el proceso hasta que el usuario decida salir.

    En este caso, todas las características del dataset son continuas,
    por lo que el modelo utiliza la parte gaussiana de Naive Bayes.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    target_names = iris.target_names

    model = NaiveBayes(alpha=1.0)
    model.fit(df, target_col="target")

    print_separator()
    print(TITLE + "NAIVE BAYES - DATASET IRIS")
    print_separator()

    print(INFO + "Resumen del modelo:")
    print(model.summary())

    print(INFO + "\nMapeo de clases:")
    for i, name in enumerate(target_names):
        print(f"  {i} -> {name}")

    while True:
        print_separator()
        print(TITLE + "CAPTURA DE NUEVA FLOR")
        print_separator()

        sample = {
            "sepal length (cm)": ask_float("sepal length (cm)"),
            "sepal width (cm)": ask_float("sepal width (cm)"),
            "petal length (cm)": ask_float("petal length (cm)"),
            "petal width (cm)": ask_float("petal width (cm)")
        }

        pred, log_scores, details = model.predict_one(sample, return_details=True)
        show_result(model, pred, log_scores, details, target_names)

        again = input(INFO + "\n¿Deseas probar otra flor? (s/n): " + RESET).strip().lower()
        if again != "s":
            print(OK + "\nPrograma terminado.")
            break


if __name__ == "__main__":
    main()