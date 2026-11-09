"""
Problema 4: Manejo de datos en una tabla de sensores
-----------------------------------------------------
Matriz 5x5 de temperaturas (°C):
    filas    -> momentos en el tiempo (08:00 ... 16:00)
    columnas -> sensores (S1 ... S5)

Con un arreglo 2D las estadísticas por columna y por fila son la MISMA operación
cambiando el eje:  axis=0 recorre las filas (resultado por sensor) y
                   axis=1 recorre las columnas (resultado por momento).
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import guardar_figura, encabezado

SENSORES = ["S1", "S2", "S3", "S4", "S5"]
MOMENTOS = ["08:00", "10:00", "12:00", "14:00", "16:00"]

# S4 está junto a una ventana con sol directo: por eso varía mucho más que los demás
TEMPERATURAS = np.array([
    [21.2, 21.5, 20.9, 22.1, 21.0],
    [22.4, 22.8, 22.0, 24.9, 22.3],
    [24.1, 24.5, 23.8, 28.7, 24.0],
    [25.3, 25.6, 24.9, 30.2, 25.1],
    [23.8, 24.1, 23.5, 27.4, 23.6],
])


def estadisticas(tabla, eje):
    """Media y desviación estándar muestral (ddof=1) a lo largo de un eje."""
    return tabla.mean(axis=eje), tabla.std(axis=eje, ddof=1)


def imprimir_tabla(tabla, media_fila, desv_fila, media_col, desv_col):
    """Tabla con los datos y sus estadísticas en los márgenes."""
    print(f"\n{'Hora':>6} | " + " ".join(f"{s:>6}" for s in SENSORES) + " |  Media   Desv.")
    print("-" * 64)
    for momento, fila, m, d in zip(MOMENTOS, tabla, media_fila, desv_fila):
        print(f"{momento:>6} | " + " ".join(f"{v:6.1f}" for v in fila) + f" | {m:6.2f} {d:6.2f}")
    print("-" * 64)
    print(f"{'Media':>6} | " + " ".join(f"{v:6.2f}" for v in media_col))
    print(f"{'Desv.':>6} | " + " ".join(f"{v:6.2f}" for v in desv_col))


def graficar(tabla, media_col, desv_col, media_fila, desv_fila):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3))

    # 1) Mapa de calor de la tabla completa
    im = a1.imshow(tabla, cmap="YlOrRd")
    a1.set(xticks=range(5), xticklabels=SENSORES, yticks=range(5), yticklabels=MOMENTOS,
           title="Temperaturas (°C)")
    for i in range(5):
        for j in range(5):
            a1.text(j, i, f"{tabla[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=a1, fraction=0.046)

    # 2) Por sensor (columnas): media ± desviación
    a2.bar(SENSORES, media_col, yerr=desv_col, capsize=6,
           color=["#e67e22" if s == np.argmax(desv_col) else "#3b6ea5" for s in range(5)])
    a2.set(title="Por sensor (columnas): media ± desv.", ylabel="°C", ylim=(18, 31))

    # 3) Por momento (filas): media con banda de ± desviación
    x = np.arange(5)
    a3.plot(x, media_fila, marker="o", color="#3b6ea5", label="Media")
    a3.fill_between(x, media_fila - desv_fila, media_fila + desv_fila,
                    alpha=0.25, color="#3b6ea5", label="± desviación")
    a3.set(xticks=x, xticklabels=MOMENTOS, title="Por momento (filas)", ylabel="°C", ylim=(18, 31))
    a3.legend()
    fig.tight_layout()
    return guardar_figura(fig, "problema4_sensores")


def main():
    encabezado("PROBLEMA 4: Tabla de sensores de temperatura 5x5")
    media_col, desv_col = estadisticas(TEMPERATURAS, eje=0)     # por sensor
    media_fila, desv_fila = estadisticas(TEMPERATURAS, eje=1)   # por momento
    imprimir_tabla(TEMPERATURAS, media_fila, desv_fila, media_col, desv_col)

    s = int(np.argmax(desv_col))
    m = int(np.argmax(media_fila))
    print(f"\nSensor más variable: {SENSORES[s]} (σ = {desv_col[s]:.2f} °C, "
          f"{desv_col[s] / np.median(desv_col):.1f} veces la mediana de los demás)")
    print(f"Momento más caluroso: {MOMENTOS[m]} (media {media_fila[m]:.2f} °C); "
          f"su desviación entre sensores ({desv_fila[m]:.2f} °C) es la mayor por efecto de {SENSORES[s]}")
    graficar(TEMPERATURAS, media_col, desv_col, media_fila, desv_fila)
    return media_col, desv_col, media_fila, desv_fila


if __name__ == "__main__":
    main()
