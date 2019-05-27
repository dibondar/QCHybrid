"""
Plotting the analytical solutions for the paper (see Fig. 1)
"""
from plot_analytic_solution import CAnalyticQCHybrid
from solution_ageq import SolAGKEq
from wigner_normalize import WignerNormalize, WignerSymLogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

"""
# Initialize the hybrid solution
p = np.linspace(-0.1, 0.1, 500)[:, np.newaxis]
q = np.linspace(-0.1, 0.1, 500)[np.newaxis, :]

hybrid = CAnalyticQCHybrid(p, q, kT=1e-5)
agk = SolAGKEq(p, q, kT=1e-5)
"""
# Initialize the hybrid solution
p = np.linspace(-0.1, 0.1, 1000)[:, np.newaxis]
q = np.linspace(-0.1, 0.1, 1000)[np.newaxis, :]

params = {
    "p":p,
    "q":q,
    "omega":1.,
    "beta":1e5,
    "alpha":0.95,
}
hybrid = CAnalyticQCHybrid(**params)
agk = SolAGKEq(**params)


# calculate quantum purity
#time = np.linspace(0., 14, 1500)
#time = np.linspace(0., 14, 150)
time = np.linspace(0., 30000, 30)

pauli_matrices = [
    np.array([[0., 1.], [1., 0.]]),
    np.array([[0., -1.j], [1.j, 0.]]),
    np.array([[1., 0], [0., -1]])
]

quantum_purity = []
quantum_state_points = []

agk_quantum_purity = []
agk_quantum_state_points = []

agk_rho0 = agk.calculate_D(0).quantum_density()


for t in time:
    ####################################################################################################################
    #
    #   Our hybrid (the exact solution)
    #
    ####################################################################################################################

    rho = hybrid.calculate_D(t).quantum_density()

    # save the purity of the state
    quantum_purity.append(rho.dot(rho).trace().real)

    quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )

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

    if not np.allclose(hybrid.classical_density(), agk.classical_density()):
        print("There is difference between classical distributions")

    if not np.allclose(agk_rho0, rho):
        print("I am wrong!!!")
        print("t = ", t)
        print(np.abs(agk_rho0 - rho).max())
        print("\n")

quantum_purity = np.array(quantum_purity)
agk_quantum_purity = np.array(agk_quantum_purity)

quantum_state_points = np.array(quantum_state_points)
agk_quantum_state_points = np.array(agk_quantum_state_points)

# Calculate quantities to be plotted
classical_densities = {}
agk_classical_densities = {}

quantum_rho = {}
agk_quantum_rho = {}

#time_slices = [("(a)", 0), ("(b)", 2.4), ("(c)", 5.7), ("(d)", 8.8)]
time_slices = [("(a)", 0), ("(b)", 0.5), ("(c)", 1.5), ("(d)", 14)]

purity_time_slices = []
agk_purity_time_slices = []

for label, t in time_slices:

    ####################################################################################################################
    #
    #   Our hybrid (the exact solution)
    #
    ####################################################################################################################

    # calculate the hybrid density matrix
    hybrid.calculate_D(t)

    # save the classical denisty and quantum state
    classical_densities[label] = hybrid.classical_density()

    rho = hybrid.quantum_density()
    quantum_rho[label] = [sigma.dot(rho).trace().real for sigma in pauli_matrices]

    purity_time_slices.append(
        (t, rho.dot(rho).trace().real)
    )

    ####################################################################################################################
    #
    #   Aleksandrov-Gerasimenko-Kapral Eq exact solution
    #
    ####################################################################################################################

    agk.calculate_D(t)
    agk_classical_densities[label] = agk.classical_density()

    rho = agk.quantum_density()
    agk_quantum_rho[label] = [sigma.dot(rho).trace().real for sigma in pauli_matrices]

    agk_purity_time_slices.append(
        (t, rho.dot(rho).trace().real)
    )

#########################################################################
#
# Fig
#
#########################################################################

img_params = dict(
    extent=[q.min(), q.max(), p.min(), p.max()],
    origin='lower',
    cmap='seismic',
    #norm=WignerNormalize(vmin=-0.1, vmax=0.1)
    norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.01, vmax=0.1)
)

plt.subplot(221)

plt.imshow(agk_classical_densities["(a)"], **img_params)
plt.text(0.2, 0.8, "(a)", transform=plt.gca().transAxes)

#plt.xlabel('$q$ (a.u.)')
plt.gca().get_xaxis().set_visible(False)

plt.ylabel('$p$ (a.u.)')


plt.subplot(222)

plt.imshow(classical_densities["(b)"], **img_params)
plt.text(0.2, 0.8, "(b)", transform=plt.gca().transAxes)

# plt.xlabel('$q$ (a.u.)')
plt.gca().get_xaxis().set_visible(False)
# plt.ylabel('$p$ (a.u.)')
plt.gca().get_yaxis().set_visible(False)


plt.subplot(223)

plt.imshow(agk_classical_densities["(c)"], **img_params)
plt.text(0.2, 0.8, "(c)", transform=plt.gca().transAxes)

plt.xlabel('$q$ (a.u.)')
plt.ylabel('$p$ (a.u.)')


plt.subplot(224)

plt.imshow(agk_classical_densities["(d)"], **img_params)
plt.text(0.2, 0.8, "(d)", transform=plt.gca().transAxes)

plt.xlabel('$q$ (a.u.)')
# plt.ylabel('$p$ (a.u.)')
plt.gca().get_yaxis().set_visible(False)

plt.show()

#########################################################################
#
# Bloch sphere plot
#
#########################################################################

# ax = plt.subplot(111, projection='3d')
#
# bloch_sphere = Bloch(axes=ax) #(view=[90.,0.],)
# bloch_sphere.ylpos = [0.3, -1.2]
# bloch_sphere.xlpos = [0.3, -1.07]
# #bloch_sphere.zlpos = [1.07, -1.07]
# bloch_sphere.zlabel = ['', '']
# # bloch_sphere.size = [600, 600]
# # bloch_sphere.point_size
#
# bloch_sphere.point_color = ['k',]
#
# bloch_sphere.add_points(
#     np.array(list(quantum_rho.values())).T
# )
#
# # # plot the trajectory bloch vector traces
# start_point = quantum_state_points[0]
#
# colouring = np.linspace(0, 1, len(quantum_state_points)- 1)
#
# for end_point, color_code in zip(quantum_state_points[1:], colouring):
#     # following the convention in the Bloch source code
#     ax.plot(
#         (start_point[1], end_point[1]),
#         (-start_point[0], -end_point[0]),
#         (start_point[2], end_point[2]),
#         alpha=1., zdir='z',
#         color=plt.cm.viridis(color_code),
#         linewidth=1.
#     )
#     start_point = end_point
# #
# #
# # for label, rho in quantum_rho.items():
# #     bloch_sphere.add_annotation(rho, label)
#
# bloch_sphere.show()
# plt.show()

####################################################################################################################
#
#   Our hybrid
#
####################################################################################################################

colouring = np.linspace(0, 1, len(quantum_state_points)- 1)

# plot the trajectory bloch vector traces
start_point = quantum_state_points[0]

for end_point, color_code in zip(quantum_state_points[1:], colouring):
    plt.plot(
        (start_point[1], end_point[1]),
        (start_point[2], end_point[2]),
        color=plt.cm.viridis(color_code),
    )
    start_point = end_point

# add points where the classical densities where plotted
plt.plot(
    *np.array(list(quantum_rho.values())).T[1:],
    'ok', markersize=5,
)

####################################################################################################################
#
#   Aleksandrov-Gerasimenko-Kapral
#
####################################################################################################################

colouring = np.linspace(0, 1, len(agk_quantum_state_points)- 1)

# plot the trajectory bloch vector traces
start_point = agk_quantum_state_points[0]

for end_point, color_code in zip(agk_quantum_state_points[1:], colouring):
    plt.plot(
        (start_point[1], end_point[1]),
        (start_point[2], end_point[2]),
        ':',
        color=plt.cm.viridis(color_code),
    )
    start_point = end_point

# add points where the classical densities where plotted
plt.plot(
    *np.array(list(agk_quantum_rho.values())).T[1:],
    'ok', markersize=5,
)

####################################################################################################################

# display the arch of Bloch sphere
plt.gca().add_patch(
    patches.Arc((0., 0.), 2., 2., theta1=0., theta2=180., fill=False, linestyle='--')
)

plt.xlabel('y axis of Bloch sphere, ${\\rm Tr}\, \left(\widehat{\sigma}_2 \hat{\\rho}(t) \\right)$')
plt.ylabel('z axis of Bloch sphere, ${\\rm Tr}\, \left(\widehat{\sigma}_3 \hat{\\rho}(t) \\right)$')

plt.show()

####################################################################################################################
#
# Purity plot
#
####################################################################################################################


####################################################################################################################
#
#   Our hybrid
#
####################################################################################################################

start_purity = quantum_purity[0]
start_time = time[0]

for end_time, end_purity, color_code in zip(time[1:], quantum_purity[1:], colouring):
    plt.plot(
        (start_time, end_time),
        (start_purity, end_purity),
        color=plt.cm.viridis(color_code),
        # linewidth=1.2
    )
    start_purity = end_purity
    start_time = end_time

# put points where the classical density is plotted
plt.plot(*zip(*purity_time_slices), 'ok', markersize=5)

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
        ':',
        color=plt.cm.viridis(color_code),
        # linewidth=1.2
    )
    start_purity = end_purity
    start_time = end_time

# put points where the classical density is plotted
plt.plot(*zip(*agk_purity_time_slices), 'ok', markersize=5)

####################################################################################################################

plt.xlabel("time (a.u.)")
plt.ylabel("quantum purity")
plt.show()