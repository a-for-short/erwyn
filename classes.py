"""
    Let's define what we are working with here.
    We want a following input -> output:
        input: an arbitrary shaped potential and an arbitrary shaped
            wavefunction
        output: evolution of wavefunction in time
    Time dependency is completely given by solving the stationary Schrödinger's
    equation (SSE). The steps are as follows:
        1) find eigenfunctions and eigenvalues for a potential given
            (i.e. solve SSE)
        2) find the scalar products of wavefunction given with basis
            functions
        3) animate evolving the coefficients

    SSE is:
        - h_bar^2/(2m_e)d^2 psi / (d x)^2 + V psi = E psi
    To solve it we'll be working in finite basis in coordinate representation.
    So the plan to solve SSE is as follows:
        a) discretize everything
            x -> vec(x_i)
            V -> diag(v_i)
            psi -> {psi_i}
        b) discretize second derivative
            This one is a tridiagonal matrix.
            d^2 f/ (d x^2) = (f_(i+1) - 2 f_i + f_(i-1))/(d x)^2
        c) diagonalize the Hamiltonian finding the coordinate representations
            of finite basis and their respective energies
        d) animate evolution.

    With this we can commence forth to coding.
"""
import numpy as np


class Grid:
    """Class for discrete coordinates"""

    def __init__(self, x_min, x_max, num_points):
        self.x_min = x_min
        self.x_max = x_max
        self.num_points = num_points

        self.x = np.linspace(x_min, x_max, num_points)
        self.dx = self.x[1] - self.x[0]


class Potential:
    """A class for potential well"""

    def __init__(self, grid: Grid, values):
        self.grid = grid
        self.values = values
        self.x = self.grid.x


class WaveFunction:
    """A class for wavefunction"""

    def __init__(self, grid: Grid, values):
        self.grid = grid
        self.values = values


class Hamiltonian:
    """A class for Hamiltonian"""

    def __init__(self, potential: Potential, mass=1.0, hbar=1.0):
        self.mass = mass
        self.hbar = hbar
        self.potential = potential
        self.x = potential.x
        self.grid = potential.grid
        self.num_points = potential.grid.num_points
        self.dx = potential.grid.dx

    def kinetic_energy(self):
        coefficient = -(self.hbar**2)/(2*self.mass)*(1/(self.dx**2))
        T = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            T[i, i] = -2
            if i > 0:
                T[i, i - i] = 1
            if i < self.num_points - 1:
                T[i, i + 1] = 1
        return coefficient * T

    def potential_energy(self):
        V = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            V[i, i] = self.potential.values[i]
        return V

    def matrix(self):
        return self.kinetic_energy() + self.potential_energy()

    def diagonalize(self):
        H = self.matrix()
        energies, functions = np.linalg.eigh(H)
        energies = Energies(self, energies)
        return energies, functions


class Energies:
    def __init__(self, hamiltonian, values):
        self.values = values
        self.grid = grid
        self.hamiltonian = hamiltonian

    def plot_energy_levels(self):
        plt.hlines(self.values, grid.x_min, grid.x_max)

    def plot_potential_and_energies(self):
        plt.plot(self.hamiltonian.grid.x, self.hamiltonian.potential.values)
        plt.hlines(self.values, grid.x_min, grid.x_max)


import matplotlib.pyplot as plt
grid = Grid(0, 10, 300)


def potential_function(x, x_0=1, k=1):
    return k*(x - x_0)**2/2


potential = Potential(grid,
                      potential_function(grid.x,
                                         x_0=(grid.x_max+grid.x_min)/2, k=2)
                      )

plt.plot(grid.x, potential)

H = Hamiltonian(potential)
E, psi = H.diagonalize()
#E.plot_potential_and_energies()
plt.plot(grid.x, psi[7])
plt.show()
