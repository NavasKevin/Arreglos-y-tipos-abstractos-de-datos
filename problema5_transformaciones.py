"""
Problema 5: Transformación de coordenadas en un sistema cartesiano
-------------------------------------------------------------------
Los puntos se guardan en un arreglo 2D de forma (N, 2): una fila por punto (x, y).

Se usan COORDENADAS HOMOGÉNEAS (x, y, 1) para que escalado, rotación y traslación
sean todas matrices 3x3 y se puedan COMPONER en una sola matriz:

    M = T · R · S      (primero escala, luego rota, al final traslada)

Aplicar M a todos los puntos es una única multiplicación de matrices.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import guardar_figura, encabezado

# Figura: contorno de una casa (el último punto repite el primero para cerrarla)
PUNTOS = np.array([[0, 0], [4, 0], [4, 3], [2, 5], [0, 3], [0, 0]], dtype=float)


def matriz_escalado(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=float)


def matriz_rotacion(grados):
    """Rotación antihoraria alrededor del origen."""
    t = np.radians(grados)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])


def matriz_traslacion(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)


def transformar(matriz, puntos):
    """Aplica una matriz 3x3 a un arreglo (N, 2) de puntos.
    Se agrega la columna de unos (homogéneas) y se multiplica todo de una vez:
    (N x 3) · (3 x 3)ᵀ -> (N x 3)."""
    homogeneas = np.hstack([puntos, np.ones((len(puntos), 1))])
    return (homogeneas @ matriz.T)[:, :2]


def area_poligono(puntos):
    """Área con la fórmula del zapato (shoelace), usando las columnas del arreglo."""
    x, y = puntos[:, 0], puntos[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def graficar(puntos, pasos, final, orden_inverso):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.fill(*puntos.T, alpha=0.25, color="gray")
    a1.plot(*puntos.T, "k-o", label="Original")
    estilos = ["#3b6ea5", "#8e44ad", "#c0392b"]
    for (nombre, pts), color in zip(pasos, estilos):
        a1.plot(*pts.T, "--", color=color, marker=".", label=nombre)
    a1.fill(*final.T, alpha=0.25, color="#c0392b")
    a1.set(title="Antes y después: escalar -> rotar -> trasladar", aspect="equal")
    a1.axhline(0, color="gray", lw=0.6); a1.axvline(0, color="gray", lw=0.6)
    a1.grid(alpha=0.3); a1.legend()

    a2.plot(*puntos.T, "k-o", label="Original")
    a2.fill(*final.T, alpha=0.3, color="#c0392b")
    a2.plot(*final.T, color="#c0392b", marker="o", label="M = T·R·S")
    a2.fill(*orden_inverso.T, alpha=0.3, color="#16a085")
    a2.plot(*orden_inverso.T, color="#16a085", marker="o", label="M = S·R·T (orden inverso)")
    a2.set(title="El orden de las matrices importa", aspect="equal")
    a2.axhline(0, color="gray", lw=0.6); a2.axvline(0, color="gray", lw=0.6)
    a2.grid(alpha=0.3); a2.legend()
    fig.tight_layout()
    return guardar_figura(fig, "problema5_transformaciones")


def main():
    encabezado("PROBLEMA 5: Transformación de coordenadas")
    S = matriz_escalado(1.5, 1.5)
    R = matriz_rotacion(30)
    T = matriz_traslacion(6, 2)
    M = T @ R @ S                                    # una sola matriz compuesta

    np.set_printoptions(precision=3, suppress=True)
    print("Puntos originales (N x 2):\n", PUNTOS[:-1])
    print("\nMatriz compuesta M = T·R·S:\n", M)

    final = transformar(M, PUNTOS)
    print("\nPuntos transformados:\n", final[:-1])

    # Verificaciones
    paso_a_paso = transformar(T, transformar(R, transformar(S, PUNTOS)))
    print(f"\n¿M aplicada de una vez = tres transformaciones sucesivas? {np.allclose(final, paso_a_paso)}")
    lado_original = np.linalg.norm(PUNTOS[1] - PUNTOS[0])
    lado_final = np.linalg.norm(final[1] - final[0])
    print(f"Longitud del lado inferior: {lado_original:.2f} -> {lado_final:.2f} "
          f"(factor {lado_final / lado_original:.2f} = escala; rotar y trasladar no deforman)")
    print(f"Área: {area_poligono(PUNTOS):.2f} -> {area_poligono(final):.2f} "
          f"(factor {area_poligono(final) / area_poligono(PUNTOS):.2f} = det(M) = {np.linalg.det(M):.2f})")

    pasos = [("1. Escalado x1.5", transformar(S, PUNTOS)),
             ("2. + Rotación 30°", transformar(R @ S, PUNTOS)),
             ("3. + Traslación (6, 2)", final)]
    graficar(PUNTOS, pasos, final, transformar(S @ R @ T, PUNTOS))
    return final


if __name__ == "__main__":
    main()
