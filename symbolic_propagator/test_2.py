########################################################################################################################
#
#   The displacement formula
#      exp(t * d_x) f(x) = f(x + t) = exp(-1j * t * (1j * d_x)) f(x)
#
########################################################################################################################

from symbolic_propagator import *
from sympy import sin, symbols
import matplotlib.pyplot as plt

x = symbols("x")
x_vals = np.linspace(0., 2. * np.pi, 200)
t = 2.

exact = 5 * np.sin((x_vals + t) ** 2)
approx = symbolic_exp(5 * sin(x ** 2), lambda _: 1j * _.diff(x), t, {"x": x_vals}, 1e-7)

print(
    "Absolute errors = {}".format(
        np.abs(approx - exact).max()
    )
)
print(
    "Relative errors = {}".format(
        np.abs(1. - approx / exact).max()
    )
)

plt.plot(x_vals, exact, label='exact')
plt.plot(x_vals, approx, label='symbolic propagation')
plt.xlabel('$x$')
plt.ylabel('')
plt.legend()
plt.show()