"""
Testing the classical quantum wavefunction formalism for the case of harmonic oscilator

"""
from quant_class_hybrid import *
from scipy.linalg import eigh

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
            norm=WignerNormalize(vmin=-0.01, vmax=0.05),
            # norm=WignerSymLogNorm(linthresh=1e-5, vmin=-0.01, vmax=0.1),
        )

        ax = fig.add_subplot(121)
        ax.set_title('Classical density, $\\rho(x,p,t)$')

        # generate empty plots
        self.img_clasical_rho = ax.imshow(
            [[]],
            **img_params,
            extent=[self.quant_sys.X.min(), self.quant_sys.X.max(), self.quant_sys.P.min(), self.quant_sys.P.max()],
        )

        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$p$ (a.u.)')

        ax = fig.add_subplot(122)
        ax.set_title('Real part quantum density matrix, $\Re\hat{\\rho}$')

        # generate empty plots
        self.img_Upsilon2 = ax.imshow([[]], **img_params, extent=[1, 2, 1, 2])

        #ax.set_xlabel('$x$ (a.u.)')
        #ax.set_ylabel('$p$ (a.u.)')

        #self.fig.colorbar(self.img_clasical_rho)

    def set_sys(self):
        """
        Initialize quantum propagator
        :param self:
        :return:
        """
        def post_initialization(self):

            self.energy = []
            self.time = []
            self.q_entropy = []
            self.q_purity = []

            self.hamiltonian_observable = (self.K + " + " + self.U, self.f1, self.f2, self.f3)

            # initialize the copy of wavefunc
            self.__Upsilon1 = np.empty_like(self.Upsilon1)
            self.__Upsilon2 = np.empty_like(self.Upsilon2)

        def post_processing(self):
            self.time.append(self.t)

            # Make a copy of the current wave function before modulating it
            np.copyto(self.__Upsilon1, self.Upsilon1)
            np.copyto(self.__Upsilon2, self.Upsilon2)

            # modulate the wave function: Upsilon *= exp(+H)
            self.exp_plus_H_Upsilon()

            self.normalize()

            # calculate the hybrid density matrix
            self.get_hybrid_D()

            # remove noise in the hybrid density matrix
            ne.evaluate("where(X ** 2 + P ** 2 > 5. ** 2, 0., D11)", local_dict=vars(self), out=self.D11)
            ne.evaluate("where(X ** 2 + P ** 2 > 5. ** 2, 0., D12)", local_dict=vars(self), out=self.D12)
            ne.evaluate("where(X ** 2 + P ** 2 > 5. ** 2, 0., D22)", local_dict=vars(self), out=self.D22)

            # renormalize the density matrix
            N = ne.evaluate("sum(D11 + D22)", local_dict=vars(self)) * self.dXdP
            self.D11 /= N
            self.D12 /= N
            self.D22 /= N

            # save the current value of the energy
            self.energy.append(
                self.hybrid_average(self.hamiltonian_observable, calculate=False)
            )

            # calculate quantum density matrix
            rho12 = self.D12.sum()
            self.quantum_rho = np.array(
                [[self.D11.sum(), rho12],
                 [rho12.conj(), self.D22.sum()]]
            )
            self.quantum_rho /= self.quantum_rho.trace()

            # calculate eigenvalues of the quantum density matrix
            p = eigh(self.quantum_rho, eigvals_only=True)

            # Check that quantum density matrix is positively defined
            assert np.allclose(np.where(p < 0, p, 0), 0), \
                "Quantum density matrix must be positively defined {}".format(p)

            np.abs(p, out=p)

            assert np.isclose(p.sum(), 1), "Trace of the quantum density matrix must be 1"

            # save the entropy of the quantum density matrix
            self.q_entropy.append(
                -np.sum(p * np.log(p + 1e-100))
            )

            # save the purity of the quantum density matrix
            self.q_purity.append(
               np.sum(p ** 2)
            )

            # restore the original wave function
            np.copyto(self.Upsilon1, self.__Upsilon1)
            np.copyto(self.Upsilon2, self.__Upsilon2)


        self.quant_sys = QCHybrid(
            t=0,
            dt=0.01,

            X_gridDIM=2 * 256,
            #X_amplitude=11.,
            X_amplitude=9,

            P_gridDIM=2 * 256,
            #P_amplitude=10.,
            P_amplitude=9,

            # Parameters of the harmonic oscillator
            omega=1,
            m=1,
            beta=1, # the inverse temperature used in the initial condition

            D=0.000,

            # Parameters of the Hamiltonian
            K="P ** 2 / (2. * m)",
            diff_K="P / m",

            U="0.5 * (omega * X) ** 2 ",
            diff_U="omega ** 2 * X",

            f1="0.1 * X ** 2",
            diff_f1="0.1 * 2 * X",

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
            r="sqrt((omega * sqrt(m) * X) ** 2 + P ** 2 / m)",
            theta="arctan2(P / sqrt(m), omega * sqrt(m) * X)"
        )

        quant_sys = self.quant_sys

        quant_sys.set_wavefunction(Upsilon, "0 * X + 0 * P")

        quant_sys.exp_minus_H_Upsilon()


    def __call__(self, frame_num):
        """
        Draw a new frame
        :param frame_num: current frame number
        :return: image objects
        """
        quant_sys = self.quant_sys
        quant_sys.propagate(10)

        # propagate the wigner function
        self.img_clasical_rho.set_array(
            (quant_sys.D22 + quant_sys.D11).real
            #quant_sys.get_classical_rho()
        )

        self.img_Upsilon2.set_array(
            quant_sys.quantum_rho.real
        )

        return self.img_clasical_rho, self.img_Upsilon2


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

plt.subplot(311)
plt.title("Energy")
plt.plot(quant_sys.time, quant_sys.energy)
plt.xlabel('$t$ (a.u.)')
plt.ylabel('$\\langle \hat{H} \\rangle$')

print(
    "Energy variation = {:.1e} %".format(
        100. * (1 - np.abs(quant_sys.energy).min() / np.abs(quant_sys.energy).max())
    )
)

plt.subplot(312)
plt.title("Entropy of the quantum subsystem")
plt.plot(quant_sys.time, quant_sys.q_entropy)
plt.xlabel('$t$ (a.u.)')
plt.ylabel('$-{\\rm Tr} \, (\hat{\\rho} \log\hat{\\rho}) $')

plt.subplot(313)
plt.title("Purity of the quantum subsystem")
plt.plot(quant_sys.time, quant_sys.q_purity)
plt.xlabel('$t$ (a.u.)')
plt.ylabel('${\\rm Tr} \, (\hat{\\rho}^2) $')

plt.show()