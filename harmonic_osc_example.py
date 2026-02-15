import numpy as np
import matplotlib.pyplot as plt
from core import Grid, Hamiltonian, TimeEvolution


def harmonic(x, x0, k):
    return 0.5 * k * (x - x0)**2

grid = Grid(0, 10, 200)
V = harmonic(grid.x, x0=5, k=2)
H = Hamiltonian(grid, V)
spectrum = H.diagonalize()
#spectrum.plot(n_states=15)

# Let's set a random initial state and see how it evolves in time
def gaussian_packet(x, x0, sigma, k0):
    return (
        np.exp(-(x - x0)**2 / (2 * sigma**2))
        * np.exp(1j * k0 * x)
    )

psi0 = gaussian_packet(grid.x, x0=3, sigma=0.4, k0=5)

norm = np.sqrt(np.sum(np.abs(psi0)**2) * grid.dx)
psi0 /= norm
psi0 = psi0*10

import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 5))

x = grid.x
V = H.potential

ax.plot(x, V, color="black", linewidth=2)
line, = ax.plot([], [], color="blue", linewidth=2)

ax.set_xlim(x[0], x[-1])
ax.set_ylim(0, np.max(V) + 2)

# Project initial state
coeffs = spectrum.project(psi0)
evolution = TimeEvolution(spectrum, coeffs)

def update(frame):
    t = frame * 0.02
    psi_t = evolution.psi_at(t)

    density = np.abs(psi_t)**2
    line.set_data(x, density + np.min(V))

    return line,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=400,
    interval=30,
    blit=True
)

plt.show()
