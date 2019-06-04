"""
Plotting the analytical solutions for the paper (see Fig. 2) using the Aleksandrov-Gerasimenko-Kapral Eq
"""
from solution_ageq import SolAGKEq

# take the same system as in Fig. 1
from fig_1 import params

from wigner_normalize import WignerNormalize, WignerSymLogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patches

agk = SolAGKEq(**params)

time = np.linspace(0., 1000000, 200)

pauli_matrices = [
    np.array([[0., 1.], [1., 0.]]),
    np.array([[0., -1.j], [1.j, 0.]]),
    np.array([[1., 0], [0., -1]])
]

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

colouring = np.linspace(0, 1, len(agk_quantum_state_points)- 1)

# plot the trajectory bloch vector traces
start_point = agk_quantum_state_points[0]

for end_point, color_code in zip(agk_quantum_state_points[1:], colouring):
    plt.plot(
        (start_point[1], end_point[1]),
        (start_point[2], end_point[2]),
        color=plt.cm.viridis(color_code),
        linewidth=2.
    )
    start_point = end_point

####################################################################################################################

# display the arch of Bloch sphere
plt.gca().add_patch(
    patches.Arc((0., 0.), 2., 2., theta1=0., theta2=180., fill=False, linestyle='--')
)

plt.xlabel('y axis of Bloch sphere, ${\\rm Tr}\, \left(\widehat{\sigma}_2 \hat{\\rho}(t) \\right)$')
plt.ylabel('z axis of Bloch sphere, ${\\rm Tr}\, \left(\widehat{\sigma}_3 \hat{\\rho}(t) \\right)$')

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

start_purity = agk_quantum_purity[0]
start_time = time[0]

for end_time, end_purity, color_code in zip(time[1:], agk_quantum_purity[1:], colouring):
    plt.plot(
        (start_time, end_time),
        (start_purity, end_purity),
        color=plt.cm.viridis(color_code),
        linewidth=2.
    )
    start_purity = end_purity
    start_time = end_time

####################################################################################################################

# plt.xlim((time.min(), time.max()))

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