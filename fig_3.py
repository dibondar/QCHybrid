"""
Comparing the Pauli equation with the Aleksandrov-Gerasimenko-Kapral Eq and our equation
"""

from solution_ageq import SolAGKEq
from solution_pauli import SolPauli
from plot_analytic_solution import CAnalyticQCHybrid

# take the Pauli system
from solution_pauli import pauli_params

from fig_1 import pauli_matrices, plot_bloch_trajectory, plot_purity_plot


from wigner_normalize import WignerNormalize, WignerSymLogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches

########################################################################################################################

time = np.linspace(0., 4, 150)

p = np.linspace(-20, 20, 500)[:, np.newaxis]
q = np.linspace(-20, 20, 500)[np.newaxis, :]

params = pauli_params.copy()
params['p'] = p
params['q'] = q

# agk = SolAGKEq(**params)
pauli = SolPauli(**params)
hybrid = CAnalyticQCHybrid(**params)

agk_quantum_purity = []
hybrid_quantum_purity = []
pauli_quantum_purity = []

agk_quantum_state_points = []
hybrid_quantum_state_points = []
pauli_quantum_state_points = []

for t in time:
    ####################################################################################################################
    #
    #   Aleksandrov-Gerasimenko-Kapral Eq exact solution
    #
    ####################################################################################################################

    """
    rho = agk.calculate_D(t).quantum_density()

    agk_quantum_purity.append(rho.dot(rho).trace().real)

    agk_quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )
    """

    ####################################################################################################################
    #
    #   Our hybrid exact solution
    #
    ####################################################################################################################

    rho = hybrid.calculate_D(t).quantum_density()

    hybrid_quantum_purity.append(rho.dot(rho).trace().real)

    hybrid_quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )

    # # check eigenvalues of D
    # D = np.vstack(
    #     (
    #         np.hstack((hybrid._D11, hybrid._D12)),
    #         np.hstack((hybrid._D12.conjugate(), hybrid._D22))
    #     )
    # )
    # eigenvals_D = np.linalg.eigvalsh(D)
    # if not np.allclose(eigenvals_D[eigenvals_D < 0], 0):
    #     print("t = {:f}     D is not positively definite ({:f})".format(t, eigenvals_D.min()))

    ####################################################################################################################
    #
    #   Pauli exact solution
    #
    ####################################################################################################################

    rho = pauli.quantum_density(t)

    pauli_quantum_purity.append(rho.dot(rho).trace().real)

    pauli_quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )

    # Classical densities must agree
    assert np.allclose(pauli.classical_density(t), hybrid.classical_density()), "Classical densities must agree"

agk_quantum_purity = np.array(agk_quantum_purity)
hybrid_quantum_purity = np.array(hybrid_quantum_purity)
pauli_quantum_purity = np.array(pauli_quantum_purity)

agk_quantum_state_points = np.array(agk_quantum_state_points)
hybrid_quantum_state_points = np.array(hybrid_quantum_state_points)
pauli_quantum_state_points = np.array(pauli_quantum_state_points)

####################################################################################################################
#
# Bloch plot
#
####################################################################################################################

plt.subplot(221)

#plt.text(-0.05, 0.5, '(A)', {'size': 16})

plt.title("Hybrid")
#plot_bloch_trajectory(agk_quantum_state_points, linestyle='-.')
plot_bloch_trajectory(hybrid_quantum_state_points, linestyle='-')
#plot_bloch_trajectory(pauli_quantum_state_points, linestyle=':')

####################################################################################################################
#
# Purity plot
#
####################################################################################################################

plt.subplot(222)

# plt.text(9e5, 0.75, '(B)', {'size':16})

plt.title("Hybrid")
#plot_purity_plot(time, agk_quantum_purity, linestyle='-.')
plot_purity_plot(time, hybrid_quantum_purity, linestyle='-')
#plot_purity_plot(time, pauli_quantum_purity, linestyle=':')


plt.subplot(223)

#plt.text(-0.05, 0.5, '(A)', {'size': 16})

plt.title("pauli")
#plot_bloch_trajectory(agk_quantum_state_points, linestyle='-.')
plot_bloch_trajectory(pauli_quantum_state_points, linestyle='-')


plt.subplot(224)

# plt.text(9e5, 0.75, '(B)', {'size':16})

plt.title("Pauli")
#plot_purity_plot(time, agk_quantum_purity, linestyle='-.')
plot_purity_plot(time, pauli_quantum_purity, linestyle='-')
#plot_purity_plot(time, pauli_quantum_purity, linestyle=':')



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


