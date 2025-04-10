import numpy as np
from scipy.special import hermite
from scipy.constants import hbar, m_e

def particle_in_box(n, L, x):
    psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
    return psi, psi**2

def harmonic_oscillator(n, x):
    # Normalized wavefunctions for harmonic oscillator
    Hn = hermite(n)
    alpha = m_e * omega / hbar
    psi = (alpha/np.pi)**0.25 * (1/np.sqrt(2**n * np.math.factorial(n))) * Hn(np.sqrt(alpha)*x) * np.exp(-alpha*x**2/2)
    return psi, np.abs(psi)**2

def quantum_tunneling(E, V0, width, x):
    # Simplified tunneling probability
    k1 = np.sqrt(2*m_e*E)/hbar
    k2 = np.sqrt(2*m_e*(V0 - E))/hbar
    transmission = 1 / (1 + (V0**2 * np.sinh(k2*width)**2)/(4*E*(V0 - E)))
    return transmission
