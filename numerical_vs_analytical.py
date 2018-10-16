"""
Comparing the solution from numerical propagation vs the analytical solution
"""

from quant_class_hybrid import *
from plot_analytic_solution import CAnalyticQCHybrid

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
            origin='lower',
            cmap='seismic',
            norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            #norm=WignerSymLogNorm(linthresh=1e-4, vmin=-0.01, vmax=0.1)
        )

        ax = fig.add_subplot(221)
        ax.set_title('Numerical classical density')

        # generate empty plots
        self.numeric_classical_density = ax.imshow(
            [[]],
            extent=[self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()],
            **img_params
        )

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(222)
        ax.set_title('Analytical classical density')

        # generate empty plots
        self.analytic_density = ax.imshow([[]], **img_params, extent=[self.q.min(), self.q.max(), self.p.min(), self.p.max()])

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(223)
        ax.set_title('Coordinate marginal')

        self.analytic_coordinate_dist, = ax.semilogy(
            [self.q.min(), self.q.max()], [1e-6, 1e1], label="analytic"
        )
        self.numeric_coordinate_dist, = ax.semilogy(
            [self.q.min(), self.q.max()], [1e-6, 1e1], label="numeric"
        )

        ax.legend()

        ax.set_xlabel('$q$ (a.u.)')
        ax.set_ylabel('Probability density')

        ax = fig.add_subplot(224)
        ax.set_title('Momentum marginal')

        self.analytic_momentum_dist, = ax.semilogy(
            [self.p.min(), self.p.max()], [1e-6, 1e1], label="analytic"
        )
        self.numeric_momentum_dist, = ax.semilogy(
            [self.p.min(), self.p.max()], [1e-6, 1e1], label="numeric"
        )

        ax.legend()

        ax.set_xlabel('$p$ (a.u.)')
        ax.set_ylabel('Probability density')

        # self.fig.colorbar(self.numeric_classical_density)

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

            # # Make a copy of the current wave function before modulating it
            # np.copyto(self.__Upsilon1, self.Upsilon1)
            # np.copyto(self.__Upsilon2, self.Upsilon2)
            #
            # # modulate the wave function: Upsilon *= exp(+H)
            # self.exp_plus_H_Upsilon()
            #
            # self.gaussian_filter(self.Upsilon1, 2., 2.)
            # self.gaussian_filter(self.Upsilon2, 2., 2.)
            #
            # self.normalize()
            #
            # self.gaussian_filter(self.Upsilon1, 4., 4.)
            # self.gaussian_filter(self.Upsilon2, 4., 4.)

            # self.get_hybrid_D()
            # self.gaussian_filter(self.D11, 2., 2.)

            self.energy.append(
                self.hybrid_average(self.hamiltonian_observable)
            )

            #
            # np.copyto(self.Upsilon1, self.__Upsilon1)
            # np.copyto(self.Upsilon2, self.__Upsilon2)

            # modulate the wave function: Upsilon *= exp(-H)
            # self.exp_minus_H_Upsilon()


        self.quant_sys = QCHybrid(
            t=0,
            dt=0.005,

            X_gridDIM=2 * 256,
            #X_amplitude=11.,
            X_amplitude=5.,

            P_gridDIM=2 * 256,
            #P_amplitude=10.,
            P_amplitude=5.,

            # Parameters of the harmonic oscillator
            beta=1e2, # the inverse temperature used in the initial condition

            D=0.000,

            # Parameters of the Hamiltonian
            K="0.5 * P ** 2",
            diff_K="P",

            U="0.5 * X ** 2 ",
            diff_U="X",

            f1="0.95 * 0.5 * X ** 2",
            diff_f1="0.95 * X",

            f2="0",
            diff_f2="0",

            f3="0",
            diff_f3="0",

            # add facility to calculate the expectation value of energy
            post_initialization=post_initialization,
            post_processing=post_processing,
        )

        # The equilibrium stationary state corresponding to the classical thermal
        # state for a harmonic oscillator with the inverse temperature beta
        Upsilon = "1 / ({r} ** 2 + 1e-10) * sqrt(1. - (1. + 0.5 * beta * {r} ** 2) * exp(-0.5 * beta * {r} ** 2) )".format(
            r="sqrt((X) ** 2 + P ** 2)",
            theta="arctan2(P, X)"
        )

        quant_sys = self.quant_sys

        quant_sys.set_wavefunction(Upsilon, "0 * X + 0 * P")

        # Initialize the analytical solution
        self.p = np.linspace(quant_sys.P.min(), quant_sys.P.max(), 500)[:, np.newaxis]
        self.q = np.linspace(quant_sys.X.min(), quant_sys.X.max(), 500)[np.newaxis, :]

        self.analytic_solution = CAnalyticQCHybrid(self.p, self.q, kT=1. / quant_sys.beta)

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """

        # Numerical propagator
        quant_sys = self.quant_sys

        numeric_density = quant_sys.get_classical_rho()

        self.numeric_classical_density.set_array(numeric_density)

        self.numeric_coordinate_dist.set_data(quant_sys.X.reshape(-1), numeric_density.sum(axis=0) * quant_sys.dX)
        self.numeric_momentum_dist.set_data(quant_sys.P.reshape(-1), numeric_density.sum(axis=1) * quant_sys.dP)


        # Firest: calculate the analytic hybrid density matrix
        self.analytic_solution.calculate_D(quant_sys.t)

        analytic_density = self.analytic_solution.classical_density()

        self.analytic_density.set_array(analytic_density)

        self.analytic_coordinate_dist.set_data(self.q, analytic_density.sum(axis=0) * self.analytic_solution.dq)
        self.analytic_momentum_dist.set_data(self.q, analytic_density.sum(axis=1) * self.analytic_solution.dp)

        # propagate numerically
        quant_sys.propagate(20)

        return self.numeric_classical_density, self.numeric_coordinate_dist, self.numeric_momentum_dist,\
               self.analytic_density, self.analytic_coordinate_dist, self.analytic_momentum_dist


#######################################################################################################

fig = plt.gcf()
visualizer = VisualizeHybrid(fig)
animation = FuncAnimation(
    fig, visualizer, frames=np.arange(100), repeat=True, blit=True
)
plt.show()

#######################################################################################################

# Check whether expectation value of energy is preserved

quant_sys = visualizer.quant_sys
quant_sys.energy = np.array(quant_sys.energy).real

plt.plot(quant_sys.time, quant_sys.energy)
plt.xlabel('$t$ (a.u.)')
plt.ylabel('$\\langle \hat{H} \\rangle$')

print(
    "Energy variation = {:.1e} %".format(
        100. * (1 - np.abs(quant_sys.energy).min() / np.abs(quant_sys.energy).max())
    )
)

plt.show()
