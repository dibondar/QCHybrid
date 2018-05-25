"""
Testing the classical quantum wavefunction formalism for the case of harmonic oscilator

"""
from quant_class_hybrid import *

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
            extent=[self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()],
            origin='lower',
            cmap='seismic',
            #norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            norm=WignerSymLogNorm(linthresh=1e-6, vmin=-0.01, vmax=0.1)
        )

        ax = fig.add_subplot(121)
        ax.set_title('Quantum classical hybrid, $\\Upsilon_1(x,p,t)$')

        # generate empty plots
        self.img_Upsilon1 = ax.imshow([[]], **img_params)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(122)
        ax.set_title('Quantum classical hybrid, $\\Upsilon_2(x,p,t)$')

        # generate empty plots
        self.img_Upsilon2 = ax.imshow([[]], **img_params)

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        self.fig.colorbar(self.img_Upsilon1)

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
            dt=0.01,

            X_gridDIM=2 * 256,
            #X_amplitude=11.,
            X_amplitude=11,

            P_gridDIM=2 * 256,
            #P_amplitude=10.,
            P_amplitude=10,

            # Parameters of the harmonic oscillator
            omega=1,
            m=1,
            X0=0,
            beta=1, # the inverse temperature used in the initial condition

            D=0.000,

            # Parameters of the Hamiltonian
            K="P ** 2 / (2. * m)",
            diff_K="P / m",
            #K="0.5 * P ** 2",
            #diff_K="P",

            U="0.5 * (omega * (X - X0)) ** 2 ",
            diff_U="omega ** 2 * (X - X0)",
            #U="0.5 * X ** 2",
            #diff_U="X",

            alpha=0.0,
            #alpha=0.,

            f1="0",
            diff_f1="0",

            #f2="alpha * X ** 4",
            #diff_f2="alpha * 4 * X ** 3",
            f2="alpha * X ** 3",
            diff_f2="alpha * 3 * X ** 2",

            f3="0",
            diff_f3="0",

            # add facility to calculate the expectation value of energy
            post_initialization=post_initialization,
            post_processing=post_processing,
        )

        # The equilibrium stationary state corresponding to the classical thermal
        # state for a harmonic oscillator with the inverse temperature beta
        Upsilon = "1 / ({r} ** 2 + 1e-10) * sqrt(1. - (1. + 0.5 * beta * {r} ** 2) * exp(-0.5 * beta * {r} ** 2) )".format(
            r="sqrt((omega * sqrt(m) * X) ** 2 + P ** 2 / m)",
            theta="arctan2(P / sqrt(m), omega * sqrt(m) * X)"
        )

        quant_sys = self.quant_sys

        quant_sys.set_wavefunction(Upsilon, "0 * X + 0 * P")

        quant_sys.exp_minus_H_Upsilon()
        #quant_sys.gaussian_filter(quant_sys.Upsilon1, 0.5, 0.5)
        #quant_sys.gaussian_filter(quant_sys.Upsilon2, 0.5, 0.5)
        quant_sys.normalize()

    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        quant_sys = self.quant_sys

        rho = quant_sys.D11 + quant_sys.D22
        #quant_sys.gaussian_filter(rho, 1, 1)

        #print(rho.imag.max(), rho.imag.min())

        # propagate the wigner function
        self.img_Upsilon1.set_array(
            #self.quant_sys.Upsilon2.imag
            #quant_sys.get_classical_rho()
            rho.real
            #quant_sys.D11.real
        )

        #print(quant_sys.classical_rho.sum() * quant_sys.dXdP)

        #print("\n")

        self.img_Upsilon2.set_array(
            quant_sys.Upsilon1.real
        )

        quant_sys.propagate(20)
        #quant_sys.exp_minus_H_Upsilon()

        return self.img_Upsilon1, self.img_Upsilon2


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
