import numpy as np
import matplotlib.pyplot as plt
from core import Grid, Hamiltonian


def harmonic(x, x0, k):
    return 0.5 * k * (x - x0)**2

grid = Grid(0, 10, 200)
V = harmonic(grid.x, x0=5, k=2)
H = Hamiltonian(grid, V)
spectrum = H.diagonalize()
spectrum.plot(n_states=15)
