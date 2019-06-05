"""
Plotting the analytical solutions for the paper (see Fig. 2) using the Aleksandrov-Gerasimenko-Kapral Eq
"""
from solution_ageq import SolAGKEq

# take the same system as in Fig. 1
from fig_1 import params, pauli_matrices, plot_bloch_trajectory, plot_purity_plot

from wigner_normalize import WignerNormalize, WignerSymLogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches

agk = SolAGKEq(**params)

time = np.linspace(0., 1000000, 200)

agk_quantum_purity = []
agk_quantum_state_points = []

for t in time:
    ####################################################################################################################
    #
    #   Aleksandrov-Gerasimenko-Kapral Eq exact solution
    #
    ####################################################################################################################

    rho = agk.calculate_D(t).quantum_density()

    agk_quantum_purity.append(rho.dot(rho).trace().real)

    agk_quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )

agk_quantum_purity = np.array(agk_quantum_purity)
agk_quantum_state_points = np.array(agk_quantum_state_points)

####################################################################################################################
#
#   Aleksandrov-Gerasimenko-Kapral
#
####################################################################################################################

plt.subplot(211)

plt.text(-0.05, 0.5, '(A)', {'size':16})

plot_bloch_trajectory(agk_quantum_state_points, linewidth=2.)

####################################################################################################################
#
# Purity plot
#
####################################################################################################################

plt.subplot(212)

plt.text(9e5, 0.75, '(B)', {'size':16})

####################################################################################################################
#
#   Aleksandrov-Gerasimenko-Kapral
#
####################################################################################################################

plot_purity_plot(time, agk_quantum_purity, linewidth=2.)

####################################################################################################################
#
#   Changing the stile of ticks
#
####################################################################################################################


def format_xticks(x, pos=None):

    if np.isclose(x, 0):
        return "0"

    if x < 0:
        return str(x)

    log10x = np.log10(x)
    exponent = int(np.floor(log10x))
    prefactor = 10. ** (log10x - exponent)

    if np.isclose(prefactor, 1):
        return "$10^{{{:d}}}$".format(exponent)
    else:
        return "${:.0f} \cdot 10^{{{:d}}}$".format(prefactor, exponent)


plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_xticks))

plt.xlabel("time (a.u.)")
plt.ylabel("quantum purity")
plt.show()