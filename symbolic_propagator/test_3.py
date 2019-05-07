########################################################################################################################
#
#   Solving Schrodinger equation
#
#
#
########################################################################################################################

from symbolic_propagator import *
from sympy import symbols, exp

from numba import njit # compile python
from QuantumClassicalDynamics.split_op_schrodinger1D import SplitOpSchrodinger1D # class for the split operator propagation
import matplotlib.pyplot as plt

########################################################################################################################
#
#   Numerically solving
#
########################################################################################################################

omega = 20

@njit
def v(x, t=0.):
    """
    Potential energy
    """
    return 0.5 * (omega * x) ** 2


@njit
def diff_v(x, t=0.):
    """
    the derivative of the potential energy for Ehrenfest theorem evaluation
    """
    return (omega) ** 4 * x ** 3


@njit
def k(p, t=0.):
    """
    Non-relativistic kinetic energy
    """
    return 0.5 * p ** 2


@njit
def diff_k(p, t=0.):
    """
    the derivative of the kinetic energy for Ehrenfest theorem evaluation
    """
    return p


# save parameters as a separate bundle
harmonic_osc_params = dict(
    x_grid_dim=512,
    x_amplitude=3.,
    v=v,
    k=k,

    diff_v=diff_v,
    diff_k=diff_k,

    dt=0.01,
)

#
nsteps = 1

# create the harmonic oscillator with time-independent hamiltonian
harmonic_osc = SplitOpSchrodinger1D(**harmonic_osc_params)

# set the initial condition
x = symbols("x")
init_state = exp(-(x + 0.01) ** 2)

harmonic_osc.set_wavefunction(
    lambda x_vals: lambdify(x, init_state, "numpy")(x_vals)
)

########################################################################################################################
#
#   Symbolical progation
#
########################################################################################################################

approx = symbolic_exp(
    init_state,
    lambda psi: -0.5 * psi.diff(x, x) + 0.5 * (omega * x) ** 2,
    nsteps * harmonic_osc.dt,
    {"x" : harmonic_osc.x}
)
approx = np.abs(approx) ** 2
approx /= approx.sum() * harmonic_osc.dx

########################################################################################################################
#
#   Plotting results
#
########################################################################################################################

plt.semilogy(
    harmonic_osc.x,
    np.abs(harmonic_osc.propagate(nsteps)) ** 2,
    label="exact"
)
plt.semilogy(
    harmonic_osc.x,
    approx,
    label='symbolic propagation'
)

plt.legend()
plt.xlabel('$x$ (a.u.)')
plt.ylabel('$\psi(x)$')
plt.show()