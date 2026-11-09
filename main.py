"""
Ejecuta los 5 problemas en orden y guarda las figuras en la carpeta resultados/.

Uso:
    python main.py          # todos los problemas
    python main.py 3        # solo el problema 3
"""
import sys

import problema1_estatica
import problema2_fluido3d
import problema3_imagenes_medicas
import problema4_sensores
import problema5_transformaciones

PROBLEMAS = {
    "1": problema1_estatica.main,
    "2": problema2_fluido3d.main,
    "3": problema3_imagenes_medicas.main,
    "4": problema4_sensores.main,
    "5": problema5_transformaciones.main,
}

if __name__ == "__main__":
    elegidos = sys.argv[1:] or list(PROBLEMAS)
    for numero in elegidos:
        if numero not in PROBLEMAS:
            print(f"Problema '{numero}' no existe. Opciones: {', '.join(PROBLEMAS)}")
            continue
        PROBLEMAS[numero]()
