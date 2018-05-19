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
            norm=WignerNormalize(vmin=-0.2, vmax=0.2)
            #norm=WignerSymLogNorm(linthresh=1e-7, vmin=-0.01, vmax=0.1)
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

        def post_processing(self):
            self.time.append(self.t)
            self.energy.append(
                self.hybrid_average(self.hamiltonian_observable)
            )

        self.quant_sys = QCHybrid(
            t=0,
            dt=0.01,

            X_gridDIM=4 * 256,
            X_amplitude=20.,

            P_gridDIM=4 * 256,
            P_amplitude=20.,

            # Parameters of the harmonic oscillator
            omega=1,
            m=1,
            X0=0,
            beta=1, # the inverse temperature used in the initial condition

            D=0,

            # Parameters of the Hamiltonian
            K="P ** 2 / (2. * m)",
            diff_K="P / m",
            #K="0.5 * P ** 2",
            #diff_K="P",

            U="0.5 * (omega * (X - X0)) ** 2",
            diff_U="omega ** 2 * (X - X0)",
            #U="0.5 * X ** 2",
            #diff_U="X",

            alpha=0.0,

            f1="alpha * X ** 3",
            diff_f1="alpha * 3 * X ** 2",

            f2="alpha * X",
            diff_f2="alpha",

            f3="alpha * X",
            diff_f3="alpha",

            # add facility to calculate the expectation value of energy
            post_initialization=post_initialization,
            post_processing=post_processing,
        )

        # The equilibrium stationary state corresponding to the classical thermal
        # state with the inverse temperature beta
        Upsilon = "exp(-beta * {r} ** 2 / 4. + 0.5j * {r} / omega * ({theta} * {r} - X0 * sin({theta})))".format(
             r="sqrt((omega * sqrt(m) * (X - X0)) ** 2 + P ** 2 / m)",
             theta="arctan2(omega * sqrt(m) * (X - X0), P / sqrt(m))"
        )

        quant_sys = self.quant_sys

        quant_sys.set_wavefunction(Upsilon, "0 * X + 0 * P")

        self.initUpsilon1 = quant_sys.Upsilon1.copy()

        quant_sys.Upsilon1 += self.initUpsilon1

        # quant_sys.gaussian_filter(quant_sys.Upsilon1, 1.6, 0)
        # quant_sys.normalize()
        #
        # final = quant_sys.Upsilon1.copy()
        #
        # quant_sys.rotate(final, 0.5 * np.pi)
        #
        # final /= final.sum()
        # quant_sys.Upsilon1 /= quant_sys.Upsilon1.sum()
        #
        # final += quant_sys.Upsilon1
        #
        # quant_sys.set_wavefunction(final, "0 * X + 0 * P")


    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        quant_sys = self.quant_sys

        quant_sys.Upsilon1 -= self.initUpsilon1
        quant_sys.propagate(10)
        quant_sys.Upsilon1 += self.initUpsilon1

        # self.quant_sys.get_hybrid_D()
        #
        # rho = np.array([
        #     [self.quant_sys.D11.sum(), self.quant_sys.D12.sum()],
        #     [self.quant_sys.D12.sum().conj(), self.quant_sys.D22.sum()]
        # ])
        # rho *= self.quant_sys.dXdP
        #
        # print(
        #     np.linalg.norm(rho - self.quant_sys.get_quantum_rho())
        # )


        #print( np.linalg.eigvalsh(self.quant_sys.get_quantum_rho()) )
        #print(self.quant_sys.quantum_rho.dot(self.quant_sys.quantum_rho).trace())
        #print()



        # propagate the wigner function
        self.img_Upsilon1.set_array(
            #self.quant_sys.Upsilon2.imag
            quant_sys.get_classical_rho()
        )

        # print(
        #     quant_sys.classical_average("X", False) - quant_sys.hybrid_average(("X", "0.", "0.", "0."), False)
        # )
        #
        # sigma_x = ((1, 0), (0, -1))
        #
        # print(
        #     quant_sys.quantum_average(sigma_x) - quant_sys.hybrid_average(("0", "0", "0", "1"), False)
        #
        # )

        print(quant_sys.classical_rho.sum() * quant_sys.dXdP)

        print("\n")

        self.img_Upsilon2.set_array(
            quant_sys.Upsilon1.imag
        )

        #avX1 = evaluate("sum(classical_rho * P)", local_dict=vars(self.quant_sys)) * self.quant_sys.dXdP
        #avX2 = self.quant_sys.classical_average("P")

        #print(avX1 / avX2)

        #print(self.quant_sys.classical_average("1"))
        #print(self.quant_sys.classical_rho.sum() * self.quant_sys.dXdP)

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
