"""
Plotting the analytical solution
"""
from plot_analytic_solution import CAnalyticQCHybrid
from wigner_normalize import WignerNormalize, WignerSymLogNorm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch

# Initialize the hybrid solution
p = np.linspace(-0.1, 0.1, 500)[:, np.newaxis]
q = np.linspace(-0.1, 0.1, 500)[np.newaxis, :]
hybrid = CAnalyticQCHybrid(p, q, kT=1e-5)

# calculate quantum purity
time = np.linspace(0., 14, 1500)

pauli_matrices = [
    np.array([[0., 1.], [1., 0.]]),
    np.array([[0., -1.j], [1.j, 0.]]),
    np.array([[1., 0], [0., -1]])
]

quantum_purity = []
quantum_state_points = []

for t in time:
    rho = hybrid.calculate_D(t).quantum_density()

    # save the purity of the state
    quantum_purity.append(rho.dot(rho).trace().real)

    quantum_state_points.append(
        [sigma.dot(rho).trace().real for sigma in pauli_matrices]
    )

quantum_purity = np.array(quantum_purity)
quantum_state_points = np.array(quantum_state_points)

# Calculate quantities to be plotted
classical_densities = dict()
quantum_rho = dict()

time_slices = [("(a)", 0), ("(b)", 2.4), ("(c)", 5.7), ("(d)", 8.8)]
purity_time_slices = []

for label, t in time_slices:

    # calculate the hybrid density matrix
    hybrid.calculate_D(t)

    # save the classical denisty and quantum state
    classical_densities[label] = hybrid.classical_density()

    rho = hybrid.quantum_density()
    quantum_rho[label] = [sigma.dot(rho).trace().real for sigma in pauli_matrices]

    purity_time_slices.append(
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
    # norm=WignerNormalize(vmin=-0.1, vmax=0.1)
    norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.01, vmax=0.1)
)

plt.subplot(221)

plt.imshow(classical_densities["(a)"], **img_params)
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

plt.imshow(classical_densities["(c)"], **img_params)
plt.text(0.2, 0.8, "(c)", transform=plt.gca().transAxes)

plt.xlabel('$q$ (a.u.)')
plt.ylabel('$p$ (a.u.)')


plt.subplot(224)

plt.imshow(classical_densities["(d)"], **img_params)
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

ax = plt.subplot(111, projection='3d')

bloch_sphere = Bloch(axes=ax) #(view=[90.,0.],)
bloch_sphere.ylpos = [0.3, -1.2]
bloch_sphere.xlpos = [0.3, -1.07]
#bloch_sphere.zlpos = [1.07, -1.07]
bloch_sphere.zlabel = ['', '']
# bloch_sphere.size = [600, 600]
# bloch_sphere.point_size

bloch_sphere.point_color = ['k',]

bloch_sphere.add_points(
    np.array(list(quantum_rho.values())).T
)

# # plot the trajectory bloch vector traces
start_point = quantum_state_points[0]

colouring = np.linspace(0, 1, len(quantum_state_points)- 1)

for end_point, color_code in zip(quantum_state_points[1:], colouring):
    # following the convention in the Bloch source code
    ax.plot(
        (start_point[1], end_point[1]),
        (-start_point[0], -end_point[0]),
        (start_point[2], end_point[2]),
        alpha=1., zdir='z',
        color=plt.cm.viridis(color_code),
        linewidth=1.
    )
    start_point = end_point
#
#
# for label, rho in quantum_rho.items():
#     bloch_sphere.add_annotation(rho, label)

bloch_sphere.show()
plt.show()

#########################################################################
#
# Purity plot
#
#########################################################################

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

plt.xlabel("time (a.u.)")
plt.ylabel("quantum purity")
plt.show()