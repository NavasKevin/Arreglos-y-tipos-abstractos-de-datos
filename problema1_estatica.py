"""
Problema 1: Análisis de fuerzas en una estructura
--------------------------------------------------
La estructura es una losa rígida con 9 nodos dispuestos en una cuadrícula 3x3
separados 2 m. La MATRIZ FUERZAS[i][j] guarda la carga vertical (kN, hacia abajo)
aplicada en el nodo de la fila i, columna j.

La losa descansa sobre 3 columnas (apoyos) no alineadas, por lo que el problema es
ESTÁTICAMENTE DETERMINADO: hay 3 reacciones desconocidas y 3 ecuaciones de equilibrio.

    ΣFz = 0   ->  R1 + R2 + R3            = Σ F
    ΣMy = 0   ->  R1·x1 + R2·x2 + R3·x3   = Σ F·x
    ΣMx = 0   ->  R1·y1 + R2·y2 + R3·y3   = Σ F·y

El sistema se escribe en forma matricial A·R = b y se resuelve con numpy.linalg.solve.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import guardar_figura, encabezado

SEPARACION = 2.0                          # distancia entre nodos (m)

# Matriz de fuerzas 3x3 (kN): cada celda es la carga aplicada en un nodo
FUERZAS = np.array([[10.0, 15.0, 10.0],
                    [20.0, 40.0, 20.0],
                    [10.0, 15.0, 30.0]])

# Apoyos: posiciones (fila, columna) de las 3 columnas que sostienen la losa
APOYOS = [(0, 0), (0, 2), (2, 1)]


def coordenadas_nodos(forma, separacion):
    """Devuelve dos matrices X, Y con la posición (m) de cada nodo.
    X[i][j] = j·s (columna) y Y[i][j] = i·s (fila)."""
    filas, columnas = forma
    return np.meshgrid(np.arange(columnas) * separacion, np.arange(filas) * separacion)


def calcular_resultante(fuerzas, X, Y):
    """Fuerza total y su punto de aplicación (centro de cargas).
    Las operaciones son elemento a elemento sobre matrices completas (sin bucles)."""
    total = fuerzas.sum()
    x_c = (fuerzas * X).sum() / total
    y_c = (fuerzas * Y).sum() / total
    return total, x_c, y_c


def calcular_reacciones(fuerzas, apoyos, separacion=SEPARACION):
    """Resuelve el sistema de equilibrio A·R = b y devuelve las 3 reacciones (kN)."""
    X, Y = coordenadas_nodos(fuerzas.shape, separacion)
    xa = np.array([X[f, c] for f, c in apoyos])
    ya = np.array([Y[f, c] for f, c in apoyos])

    # Matriz de coeficientes 3x3: una fila por ecuación de equilibrio
    A = np.vstack([np.ones(3), xa, ya])
    b = np.array([fuerzas.sum(), (fuerzas * X).sum(), (fuerzas * Y).sum()])

    if abs(np.linalg.det(A)) < 1e-9:
        raise ValueError("Los apoyos están alineados: el sistema no tiene solución única.")
    return np.linalg.solve(A, b), A, b


def graficar(fuerzas, apoyos, reacciones, resultante):
    X, Y = coordenadas_nodos(fuerzas.shape, SEPARACION)
    total, x_c, y_c = resultante
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Losa y nodos (el tamaño del círculo es proporcional a la carga)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), X.max() + 1, Y.max() + 1,
                               color="#dfe7ef", zorder=0))
    ax.scatter(X, Y, s=fuerzas * 25, color="#3b6ea5", alpha=0.75, zorder=2)
    for i in range(fuerzas.shape[0]):
        for j in range(fuerzas.shape[1]):
            ax.annotate(f"{fuerzas[i, j]:.0f} kN", (X[i, j], Y[i, j]),
                        textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9)

    # Apoyos y sus reacciones
    for (f, c), r in zip(apoyos, reacciones):
        ax.scatter(X[f, c], Y[f, c], marker="^", s=380, color="#c0392b", zorder=3)
        ax.annotate(f"R = {r:.2f} kN", (X[f, c], Y[f, c]), textcoords="offset points",
                    xytext=(0, -26), ha="center", fontsize=10, color="#c0392b", weight="bold")

    ax.scatter(x_c, y_c, marker="X", s=160, color="black", zorder=4,
               label=f"Resultante {total:.0f} kN en ({x_c:.2f}, {y_c:.2f}) m")
    ax.set(xlim=(-1, X.max() + 1), ylim=(Y.max() + 1, -1), aspect="equal",
           xlabel="x (m)", ylabel="y (m)", title="Cargas en la losa 3x3 y reacciones en los apoyos")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22))
    return guardar_figura(fig, "problema1_reacciones")


def main():
    encabezado("PROBLEMA 1: Análisis de fuerzas en una estructura 3x3")
    print("Matriz de fuerzas (kN):\n", FUERZAS)

    X, Y = coordenadas_nodos(FUERZAS.shape, SEPARACION)
    resultante = calcular_resultante(FUERZAS, X, Y)
    reacciones, A, b = calcular_reacciones(FUERZAS, APOYOS)

    print(f"\nResultante: {resultante[0]:.1f} kN aplicada en "
          f"x = {resultante[1]:.3f} m, y = {resultante[2]:.3f} m")
    print("\nSistema de equilibrio A·R = b:")
    for fila, valor in zip(A, b):
        print(f"  {fila}  ->  {valor:8.2f}")
    print("\nReacciones en los apoyos:")
    for (f, c), r in zip(APOYOS, reacciones):
        aviso = "  (negativa: el apoyo trabaja a tracción)" if r < 0 else ""
        print(f"  Nodo ({f}, {c}) en ({X[f, c]:.0f}, {Y[f, c]:.0f}) m: R = {r:7.2f} kN{aviso}")

    residuo = np.abs(A @ reacciones - b).max()
    print(f"\nVerificación del equilibrio: ΣR = {reacciones.sum():.2f} kN = ΣF = "
          f"{FUERZAS.sum():.2f} kN | residuo máximo = {residuo:.2e}")
    graficar(FUERZAS, APOYOS, reacciones, resultante)
    return reacciones


if __name__ == "__main__":
    main()
