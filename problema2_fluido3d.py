"""
Problema 2: Simulación de un fluido en una cuadrícula 3D
---------------------------------------------------------
El estado del fluido se guarda en UN arreglo de 4 dimensiones:

    estado[x, y, z, campo]   con forma (3, 3, 3, 5)

donde 'campo' es: 0 = presión (Pa), 1 = temperatura (K), 2-4 = velocidad (vx, vy, vz) en m/s.
Así cada celda del volumen 3x3x3 contiene toda su información en posiciones contiguas.

Modelo físico (acústica lineal):
  - Presión: ecuación de onda discreta. Cada celda se actualiza con el Laplaciano,
    que usa sus 6 vecinas directas (±x, ±y, ±z):
        p(t+1) = 2·p(t) - p(t-1) + (c·dt)² · ∇²p
  - Velocidad: ecuación de Euler linealizada  dv/dt = -(1/ρ)·∇p
    (el fluido acelera de las zonas de alta presión hacia las de baja presión)
  - Temperatura: compresión adiabática  T = T0 · (p/p0)^((γ-1)/γ)
  - Paredes rígidas: la onda se refleja (la vecina fuera del borde es la propia celda)
    y la velocidad perpendicular a cada pared es cero (no hay flujo a través de ella).
  - El pulso inicial se aplica en una esquina para que la onda cruce el cubo en diagonal.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import guardar_figura, encabezado

# Índices de los campos dentro de cada celda
P, T, VX, VY, VZ = range(5)

N = 3                    # celdas por lado
DX = 0.1                 # tamaño de celda (m)
P0, T0 = 101_325.0, 293.15
RHO, C, GAMMA = 1.2, 343.0, 1.4          # aire: densidad, velocidad del sonido, cp/cv
COURANT = 0.5                             # c·dt/dx; estable si <= 1/√3 ≈ 0.577 en 3D
DT = COURANT * DX / C
PULSO = 1_000.0                           # Pa de sobrepresión inicial en la esquina (0,0,0)
PASOS = 60


def crear_estado(n=N):
    """Crea el arreglo 4D en reposo y aplica un pulso de presión en una esquina."""
    estado = np.zeros((n, n, n, 5))
    estado[..., P] = P0
    estado[..., T] = T0
    estado[0, 0, 0, P] += PULSO
    estado[..., T] = temperatura_adiabatica(estado[..., P])
    return estado


def temperatura_adiabatica(presion):
    """Temperatura de cada celda tras una compresión/expansión adiabática."""
    return T0 * (presion / P0) ** ((GAMMA - 1) / GAMMA)


def laplaciano(campo, dx):
    """∇² de un campo 3D usando las 6 celdas vecinas de cada celda.
    np.pad(mode='edge') replica el borde: equivale a una pared rígida."""
    c = np.pad(campo, 1, mode="edge")
    vecinas = (c[2:, 1:-1, 1:-1] + c[:-2, 1:-1, 1:-1] +
               c[1:-1, 2:, 1:-1] + c[1:-1, :-2, 1:-1] +
               c[1:-1, 1:-1, 2:] + c[1:-1, 1:-1, :-2])
    return (vecinas - 6 * campo) / dx ** 2


def avanzar(estado, presion_anterior):
    """Actualiza TODAS las celdas un paso de tiempo a partir de sus vecinas.
    Modifica 'estado' en el lugar y devuelve la presión del paso que termina
    (la ecuación de onda necesita los dos pasos previos)."""
    p = estado[..., P].copy()
    estado[..., P] = 2 * p - presion_anterior + (C * DT) ** 2 * laplaciano(p - P0, DX)

    # Velocidad: un gradiente 3D por cada eje (x, y, z)
    for eje, gradiente in enumerate(np.gradient(estado[..., P], DX)):
        estado[..., VX + eje] -= DT / RHO * gradiente

    # Paredes rígidas: velocidad normal nula en las caras del cubo
    estado[[0, -1], :, :, VX] = 0
    estado[:, [0, -1], :, VY] = 0
    estado[:, :, [0, -1], VZ] = 0

    estado[..., T] = temperatura_adiabatica(estado[..., P])
    return p


def simular(pasos=PASOS):
    """Ejecuta la simulación y registra la presión de celdas representativas
    a lo largo del camino de la onda: esquina del pulso -> centro -> esquina opuesta."""
    estado = crear_estado()
    anterior = estado[..., P].copy()           # fluido inicialmente en reposo
    tipos = {"Pulso (0,0,0)": (0, 0, 0), "Vecina (1,0,0)": (1, 0, 0),
             "Centro (1,1,1)": (1, 1, 1), "Opuesta (2,2,2)": (2, 2, 2)}
    historia = {nombre: [estado[idx + (P,)] - P0] for nombre, idx in tipos.items()}
    instantaneas = {0: estado[..., P].copy()}
    media_inicial = estado[..., P].mean()

    for paso in range(1, pasos + 1):
        anterior = avanzar(estado, anterior)
        for nombre, idx in tipos.items():
            historia[nombre].append(estado[idx + (P,)] - P0)
        if paso in (2, 4):
            instantaneas[paso] = estado[..., P].copy()

    return estado, historia, instantaneas, media_inicial


def graficar(historia, instantaneas):
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(2, 1, 1)
    tiempo_ms = np.arange(len(next(iter(historia.values())))) * DT * 1000
    for nombre, valores in historia.items():
        ax.plot(tiempo_ms, valores, marker=".", label=nombre)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(xlabel="Tiempo (ms)", ylabel="Sobrepresión (Pa)",
           title="Propagación y reflexión de la onda de presión en el cubo 3x3x3")
    ax.legend(ncol=4, fontsize=9)

    # Capas z = 0, 1, 2 en tres instantes
    lim = max(abs(v - P0).max() for v in instantaneas.values())
    for fila, (paso, presion) in enumerate(instantaneas.items()):
        for z in range(N):
            sub = fig.add_subplot(2, 9, 10 + fila * 3 + z)
            sub.imshow(presion[:, :, z] - P0, cmap="coolwarm", vmin=-lim, vmax=lim)
            sub.set_xticks([]); sub.set_yticks([])
            sub.set_title(f"paso {paso}\ncapa z={z}", fontsize=8)
    fig.tight_layout()
    fig.text(0.5, -0.01, "Capas del cubo: rojo = sobrepresión, azul = depresión, gris = presión ambiente",
             ha="center", fontsize=9)
    return guardar_figura(fig, "problema2_fluido3d")


def main():
    encabezado("PROBLEMA 2: Simulación de fluido en una cuadrícula 3D (3x3x3)")
    estado, historia, instantaneas, media_inicial = simular()
    print(f"Arreglo de estado: forma {estado.shape} -> (x, y, z, [p, T, vx, vy, vz])")
    print(f"dt = {DT * 1e6:.1f} µs | número de Courant c·dt/dx = {COURANT} (límite de estabilidad 3D ≈ 0.577)")

    print("\nSobrepresión (Pa) en los primeros pasos:")
    print("  Paso  " + "  ".join(f"{n:>16}" for n in historia))
    for paso in range(7):
        print(f"  {paso:4}  " + "  ".join(f"{historia[n][paso]:16.1f}" for n in historia))

    print("\nEstado completo de algunas celdas al final de la simulación:")
    for nombre, idx in {"Pulso": (0, 0, 0), "Vecina": (1, 0, 0), "Centro": (1, 1, 1)}.items():
        p, t, vx, vy, vz = estado[idx]
        print(f"  {nombre:6} {idx}: p = {p:10.1f} Pa | T = {t:.4f} K | v = ({vx:+.3f}, {vy:+.3f}, {vz:+.3f}) m/s")

    variacion = abs(estado[..., P].mean() - media_inicial)
    print(f"\nConservación: la presión media varió {variacion:.2e} Pa en {PASOS} pasos "
          f"(paredes rígidas: la energía no sale del volumen).")
    graficar(historia, instantaneas)
    return estado


if __name__ == "__main__":
    main()
