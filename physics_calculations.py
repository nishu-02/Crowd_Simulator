import numpy as np

def particle_in_box(n, L, x):
    """
    Calculate wavefunction and probability density
    n: quantum number (1,2,3...)
    L: box length
    x: position array
    """
    psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
    probability = psi**2
    return psi, probability
