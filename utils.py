"""Utilidades compartidas: carpeta de resultados y guardado de figuras."""
import os

import matplotlib

matplotlib.use("Agg")                   # genera imágenes sin abrir ventanas
import matplotlib.pyplot as plt         # noqa: E402

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")


def guardar_figura(fig, nombre):
    """Guarda la figura en resultados/<nombre>.png y la cierra para liberar memoria."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, f"{nombre}.png")
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada en: resultados/{nombre}.png")
    return ruta


def encabezado(texto):
    """Imprime un encabezado de sección en la consola."""
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)
