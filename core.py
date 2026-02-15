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
import matplotlib.pyplot as plt
from matplotlib import cm

class Grid:
    """Class for discrete coordinates"""

    def __init__(self, x_min, x_max, num_points):
        self.x_min = x_min
        self.x_max = x_max
        self.x = np.linspace(x_min, x_max, num_points)
        self.num_points = num_points
        self.dx = self.x[1] - self.x[0]

class Hamiltonian:
    """Class for Hamiltonian matrix construction and diagonalization"""

    def __init__(self, grid, potential, mass=1.0, hbar=1.0):
        self.grid = grid
        self.mass = mass
        self.hbar = hbar
        self.potential = potential

    def kinetic_matrix(self):
        N = self.grid.num_points
        dx = self.grid.dx

        coeff = -(self.hbar**2) / (2 * self.mass * dx**2)

        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N - 1)

        T = (
            np.diag(main_diag)
            + np.diag(off_diag, 1)
            + np.diag(off_diag, -1)
        )

        return coeff * T

    def potential_matrix(self):
        return np.diag(self.potential)

    def matrix(self):
        return self.kinetic_matrix() + self.potential_matrix()

    def diagonalize(self):
        H = self.matrix()
        energies, eigenvectors = np.linalg.eigh(H)
        return Spectrum(self, energies, eigenvectors)

class Spectrum:
    """Class for storing and plotting the spectrum"""

    def __init__(self, hamiltonian, energies, eigenvectors):
        self.hamiltonian = hamiltonian
        self.energies = energies
        self.eigenvectors = eigenvectors
        self.grid = hamiltonian.grid

    def normalized_state(self, n):
        psi = self.eigenvectors[:, n]
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.grid.dx)
        return psi / norm

    def plot(self, n_states=5, scale=1.0, cmap_name="viridis"):
        x = self.grid.x
        V = self.hamiltonian.potential

        cmap = cm.get_cmap(cmap_name, n_states)
        colors = cmap(np.linspace(0, 1, n_states))

        plt.figure(figsize=(9, 5))

        # Potential in neutral tone
        plt.plot(x, V, color="black", linewidth=2, label="Potential")

        for n in range(n_states):
            psi = self.normalized_state(n)
            color = colors[n]

            plt.plot(x, self.energies[n] + scale * psi,
                    color=color, linewidth=2)

            plt.hlines(self.energies[n],
                    x[0], x[-1],
                    colors=color,
                    linestyles="dashed",
                    alpha=0.6)

        plt.xlabel("x")
        plt.ylabel("Energy")
        plt.title("Stationary Schrödinger Equation")
        plt.tight_layout()
        plt.show()