########################################################################################################################
#
#   The simplest test of the symbolic propagator:
#
#   Evaluate numerical exponent np.exp(-1j * vals)
#
########################################################################################################################

from symbolic_propagator import *
from sympy import symbols

vals = np.linspace(0.5, 25, 5) #20 * np.random.rand(3)
exact = np.exp(-1j * vals)

x = symbols("x")
approx = symbolic_exp(Integer(1), lambda _: _ * x, 1, {"x": vals})

print("Absolute errors = {}".format(np.abs(approx - exact)))
print("Relative errors = {}".format(np.abs(1. - approx / exact)))