"""
Below is an example when Pauli and Hybrid dynamics differ by starting from the same initial condition.
"""
from quant_class_hybrid import *
from QuantumClassicalDynamics.split_op_pauli_like1D import SplitOpPauliLike1D

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class VisualizeHybrid:
    """
    Class to visualize the phase space dynamics in phase space.
    """
    def __init__(self, fig):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """
        #  Initialize the system
        self.set_sys()

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        self.fig = fig

        # import utility to visualize the wigner function
        from wigner_normalize import WignerNormalize, WignerSymLogNorm

        img_params = dict(
            extent=[self.hybrid_sys.X.min(), self.hybrid_sys.X.max(), self.hybrid_sys.P.min(), self.hybrid_sys.P.max()],
            origin='lower',
            cmap='seismic',
            #norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            norm=WignerSymLogNorm(linthresh=1e-7, vmin=-0.01, vmax=0.1)
        )

        ax = fig.add_subplot(221)
        ax.set_title('Quantum classical hybrid, $\\Upsilon_1(x,p,t)$')

        # generate empty plots
        self.img_Upsilon1 = ax.imshow([[]], **img_params)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(222)
        ax.set_title('Quantum classical hybrid, $\\Upsilon_1(x,p,t)$')

        # generate empty plots
        self.img_Upsilon2 = ax.imshow(self.hybrid_sys.Upsilon1.real, **img_params)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        self.fig.colorbar(self.img_Upsilon2)

        ax = fig.add_subplot(223)
        ax.set_title('Coordinate marginal')

        self.hybrid_coordinate_dist, = ax.semilogy(
            [self.hybrid_sys.X.min(), self.hybrid_sys.X.max()], [1e-7, 1e0], label="hybrid"
        )
        self.pauli_coordinate_dist, = ax.semilogy(
            [self.hybrid_sys.X.min(), self.hybrid_sys.X.max()], [1e-7, 1e0], label="pauli"
        )

        ax.legend()

        ax.set_xlabel('$q$ (a.u.)')
        ax.set_ylabel('Probability density')

        ax = fig.add_subplot(224)
        ax.set_title('Momentum marginal')

        self.hybrid_momentum_dist, = ax.semilogy(
            [self.hybrid_sys.P.min(), self.hybrid_sys.P.max()], [1e-7, 1e0], label="hybrid"
        )
        self.pauli_momentum_dist, = ax.semilogy(
            [self.hybrid_sys.P.min(), self.hybrid_sys.P.max()], [1e-7, 1e0], label="pauli"
        )

        ax.legend()

        ax.set_xlabel('$p$ (a.u.)')
        ax.set_ylabel('Probability density')

    def set_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        def post_initialization(self):
            self.energy = []
            self.time = []
            self.hamiltonian_observable = (self.K + " + " + self.U, self.f1, self.f2, self.f3)

            # initialize the copy of wavefunc
            self.__Upsilon1 = np.empty_like(self.Upsilon1)
            self.__Upsilon2 = np.empty_like(self.Upsilon2)

        def post_processing(self):
            self.time.append(self.t)

            self.energy.append(
                self.hybrid_average(self.hamiltonian_observable)
            )

        self.hybrid_sys = QCHybrid(
            t=0,
            dt=0.005,

            X_gridDIM=8 * 256,
            #X_amplitude=11.,
            X_amplitude=20.,

            P_gridDIM=8 * 256,
            #P_amplitude=10.,
            P_amplitude=20,

            beta=2., # the inverse temperature used in the initial condition

            D=0.0000,

            # Parameters of the Hamiltonian
            K="0.5 * P ** 2",
            diff_K="P",

            U="0.25 * 0.01 * (X - 5.) ** 4",
            diff_U="0.01 * (X - 5.) ** 3",

            f1="0.1 * X",
            diff_f1="0.1",

            f2="0",
            diff_f2="0",

            f3="0.",
            diff_f3="0",

            # add facility to calculate the expectation value of energy
            post_initialization=post_initialization,
            #post_processing=post_processing,
        )

        # The equilibrium stationary state corresponding to the classical thermal
        # state for a harmonic oscillator with the inverse temperature beta
        Upsilon = "1 / ({r} ** 2 + 1e-10) * sqrt(1. - (1. + 0.5 * beta * {r} ** 2) * exp(-0.5 * beta * {r} ** 2) )".format(
            r="sqrt(X ** 2 + P ** 2)",
            theta="arctan2(P, X)"
        )

        hybrid_sys = self.hybrid_sys

        hybrid_sys.set_wavefunction(Upsilon, "0 * X + 0 * P")

        #print(" energry = ", hybrid_sys.hybrid_average(hybrid_sys.hamiltonian_observable))

        #hybrid_sys.set_wavefunction(hybrid_sys.hamiltonian_observable[0], "0 * X + 0 * P")

        # Set the pauli propagator
        self.pauli_sys = SplitOpPauliLike1D(
            dt=0.005,
            X_gridDIM=8 * 256,
            X_amplitude=20.,

            K0="0.5 * P ** 2",
            V0="0.25 * 0.01 * (X - 5.) ** 4",
            V1="0.1 * X",
            V3="0.",
        ).set_wavefunction("exp(-0.5 * X ** 2)")


    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        hybrid_sys = self.hybrid_sys

        classical_rho = hybrid_sys.get_classical_rho()
        self.img_Upsilon1.set_array(
            classical_rho
        )

        marginal = classical_rho.sum(axis=0)
        marginal /= marginal.max()

        self.hybrid_coordinate_dist.set_data(hybrid_sys.X.reshape(-1), marginal)
        self.hybrid_momentum_dist.set_data(hybrid_sys.P.reshape(-1), classical_rho.sum(axis=1) * hybrid_sys.dP)

        self.img_Upsilon2.set_array(
            hybrid_sys.Upsilon1.real
        )

        pauli_sys = self.pauli_sys

        density = pauli_sys.coordinate_density
        density /= density.max()
        self.pauli_coordinate_dist.set_data(pauli_sys.X, density)

        hybrid_sys.propagate(10)
        pauli_sys.propagate(10)

        return self.img_Upsilon1, self.img_Upsilon2, \
               self.hybrid_coordinate_dist, self.hybrid_momentum_dist, \
               self.pauli_coordinate_dist,


#######################################################################################################

fig = plt.gcf()
visualizer = VisualizeHybrid(fig)
animation = FuncAnimation(
    fig, visualizer, frames=np.arange(100), repeat=True, blit=True
)
plt.show()

#######################################################################################################

# Check whether expectation value of energy is preserved

hybrid_sys = visualizer.hybrid_sys
hybrid_sys.energy = np.array(hybrid_sys.energy).real

plt.plot(hybrid_sys.time, hybrid_sys.energy)
plt.xlabel('$t$ (a.u.)')
plt.ylabel('$\\langle \hat{H} \\rangle$')

print(
    "Energy variation = {:.1e} %".format(
        100. * (1 - np.abs(hybrid_sys.energy).min() / np.abs(hybrid_sys.energy).max())
    )
)

plt.show()
