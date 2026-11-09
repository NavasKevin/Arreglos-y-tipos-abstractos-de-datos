"""
Problema 3: Suavizado de imágenes médicas 3D
---------------------------------------------
El volumen es un arreglo 3D con forma (capas, alto, ancho): cada capa volumen[k] es
una imagen 2D del órgano a una profundidad distinta (como los cortes de una tomografía).

Como no se usan datos de pacientes reales, se genera un volumen SINTÉTICO:
un órgano elipsoidal con una lesión esférica más brillante, al que se suma ruido gaussiano.
Tener la imagen limpia de referencia permite MEDIR la mejora (PSNR y ruido residual).

Filtro de promedio k x k aplicado a cada capa: cada píxel se reemplaza por la media
de los k² píxeles de su vecindario dentro de la misma capa.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import guardar_figura, encabezado

CAPAS, ALTO, ANCHO = 24, 128, 128
SIGMA_RUIDO = 25.0


def generar_volumen(capas=CAPAS, alto=ALTO, ancho=ANCHO, sigma=SIGMA_RUIDO, semilla=0):
    """Devuelve (volumen_limpio, volumen_ruidoso) como arreglos 3D float32."""
    z, y, x = np.mgrid[-1:1:capas * 1j, -1:1:alto * 1j, -1:1:ancho * 1j]
    volumen = np.full((capas, alto, ancho), 20.0)                       # fondo
    organo = (x / 0.75) ** 2 + (y / 0.6) ** 2 + (z / 0.95) ** 2 <= 1
    volumen[organo] = 110 + 25 * y[organo]                               # tejido con gradiente
    lesion = (x - 0.3) ** 2 + (y + 0.1) ** 2 + (z / 1.5) ** 2 <= 0.18 ** 2
    volumen[lesion] = 200                                                # lesión
    rng = np.random.default_rng(semilla)
    ruidoso = np.clip(volumen + rng.normal(0, sigma, volumen.shape), 0, 255)
    return volumen.astype(np.float32), ruidoso.astype(np.float32)


def filtro_promedio_bucles(capa, k=3):
    """Versión DIDÁCTICA con bucles anidados (una sola capa 2D).
    Se usa como referencia para verificar y comparar tiempos."""
    r = k // 2
    alto, ancho = capa.shape
    padded = np.pad(capa, r, mode="edge")
    salida = np.empty_like(capa)
    for i in range(alto):
        for j in range(ancho):
            suma = 0.0
            for di in range(k):
                for dj in range(k):
                    suma += padded[i + di, j + dj]
            salida[i, j] = suma / (k * k)
    return salida


def filtro_promedio_volumen(volumen, k=3):
    """Versión VECTORIZADA: filtra TODAS las capas a la vez.
    Se rellena solo en los ejes de alto y ancho (eje 0 sin relleno), así cada capa
    se procesa de forma independiente. El vecindario k x k se suma con k² desplazamientos
    del arreglo completo: k² operaciones en C en lugar de capas·alto·ancho·k² en Python."""
    r = k // 2
    padded = np.pad(volumen, ((0, 0), (r, r), (r, r)), mode="edge")
    alto, ancho = volumen.shape[1:]
    acumulado = np.zeros_like(volumen)
    for di in range(k):
        for dj in range(k):
            acumulado += padded[:, di:di + alto, dj:dj + ancho]
    return acumulado / (k * k)


def psnr(referencia, imagen):
    """Relación señal/ruido de pico (dB): más alto = más parecido a la referencia."""
    mse = np.mean((referencia - imagen) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def graficar(limpio, ruidoso, suavizado, capas):
    fig, ejes = plt.subplots(len(capas), 3, figsize=(10, 3.4 * len(capas)))
    for fila, k in enumerate(capas):
        for col, (img, titulo) in enumerate([(ruidoso, "Con ruido"), (suavizado, "Suavizado 3x3"),
                                             (limpio, "Referencia sin ruido")]):
            eje = ejes[fila, col]
            eje.imshow(img[k], cmap="gray", vmin=0, vmax=255)
            eje.set_title(f"{titulo} - capa {k}" +
                          (f"\nPSNR {psnr(limpio[k], img[k]):.1f} dB" if col < 2 else "\n "), fontsize=10)
            eje.axis("off")
    fig.tight_layout()
    guardar_figura(fig, "problema3_capas")

    # Perfil de intensidad a lo largo de una fila que cruza la lesión
    k, fila = capas[0], int(ALTO * 0.45)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(ruidoso[k, fila], color="#bbbbbb", lw=1, label="Con ruido")
    ax.plot(suavizado[k, fila], color="#2a6fbb", lw=1.8, label="Suavizado 3x3")
    ax.plot(limpio[k, fila], color="black", ls="--", lw=1.2, label="Referencia")
    ax.set(xlabel="Columna (píxel)", ylabel="Intensidad",
           title=f"Perfil de intensidad: capa {k}, fila {fila} (atraviesa la lesión)")
    ax.legend()
    guardar_figura(fig, "problema3_perfil")


def main():
    encabezado("PROBLEMA 3: Suavizado de imágenes médicas 3D")
    limpio, ruidoso = generar_volumen()
    print(f"Volumen: forma {ruidoso.shape} (capas, alto, ancho) = {ruidoso.size:,} vóxeles, "
          f"{ruidoso.nbytes / 1e6:.1f} MB")

    # Verificación y comparación de tiempos: bucles vs. vectorizado
    t0 = time.perf_counter()
    capa_bucles = filtro_promedio_bucles(ruidoso[CAPAS // 2])
    t_bucles = time.perf_counter() - t0
    t0 = time.perf_counter()
    suavizado = filtro_promedio_volumen(ruidoso)
    t_vector = time.perf_counter() - t0
    assert np.allclose(capa_bucles, suavizado[CAPAS // 2], atol=1e-3)
    print(f"\nBucles anidados : {t_bucles * 1000:8.1f} ms para UNA capa "
          f"(≈ {t_bucles * CAPAS:.1f} s para las {CAPAS} capas)")
    print(f"Vectorizado     : {t_vector * 1000:8.1f} ms para TODO el volumen "
          f"-> ≈ {t_bucles * CAPAS / t_vector:,.0f} veces más rápido (mismo resultado)")

    # Mejora medida contra la referencia sin ruido
    interior = (limpio > 60) & (limpio < 160)          # tejido homogéneo del órgano
    print(f"\n{'Imagen':16} {'PSNR (dB)':>10} {'Ruido en tejido (σ)':>20}")
    print(f"{'Con ruido':16} {psnr(limpio, ruidoso):10.2f} {np.std((ruidoso - limpio)[interior]):20.2f}")
    for k in (3, 5, 7):
        filtrado = suavizado if k == 3 else filtro_promedio_volumen(ruidoso, k)
        print(f"{f'Promedio {k}x{k}':16} {psnr(limpio, filtrado):10.2f} "
              f"{np.std((filtrado - limpio)[interior]):20.2f}")
    print("(Agrandar la ventana reduce el ruido, pero también difumina los bordes del órgano y de la\n"
          " lesión: aquí 5x5 da el mejor equilibrio y con 7x7 la calidad ya empeora.)")

    graficar(limpio, ruidoso, suavizado, capas=[CAPAS // 2, CAPAS // 4])
    return suavizado


if __name__ == "__main__":
    main()
