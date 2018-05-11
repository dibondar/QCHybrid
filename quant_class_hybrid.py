import numpy as np
import numexpr as ne
import pyfftw
import pickle
import warnings
from types import MethodType, FunctionType


class QCHybrid(object):
    """
    The Koopman-von Neumann formulation of hybrid classical-quantum systems
        https://arxiv.org/abs/1802.04787
    """

    def __init__(self, *, X_gridDIM, X_amplitude, P_gridDIM, P_amplitude, U, diff_U, K, diff_K,
                 f1, diff_f1, f2, diff_f2, f3, diff_f3, dt, t, **kwargs):
        """
        Constructor
        :param X_gridDIM: the coordinate grid size
        :param X_amplitude: maximum value of the coordinates
        :param P_gridDIM: the momentum grid size
        :param P_amplitude: maximum value of the momentum
        :param U: potential energy (as a string to be evaluated by numexpr) may depend on time
        :param diff_U:  the derivative of the potential energy (as a string to be evaluated by numexpr)
        :param K: the kinetic energy (as a string to be evaluated by numexpr) may depend on time
        :param diff_K: the derivative of the kinetic energy (as a string to be evaluated by numexpr)
        :param f1, diff_f1, f2, diff_f2, f3, diff_f3: coupling functions and their derivatives
        :param dt: time step
        :param t: initial value of time
        :param kwargs: other parameters
            (e.g., D is the diffusion coefficient to mitigate the velocity filamentation)
        """

        self.X_gridDIM = X_gridDIM
        self.X_amplitude = X_amplitude
        self.P_gridDIM = P_gridDIM
        self.P_amplitude = P_amplitude
        self.U = U
        self.diff_U = diff_U
        self.K = K
        self.diff_K = diff_K

        self.f1 = f1
        self.diff_f1 = diff_f1
        self.f2 = f2
        self.diff_f2 = diff_f2
        self.f3 = f3
        self.diff_f3 = diff_f3

        self.dt = dt
        self.t = t

        # save all the other attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
            else:
                setattr(self, name, value)

        try:
            self.D
        except AttributeError:
            self.D = 0.
            warnings.warn("the diffusion coefficient (D) to mitigate the velocity filamentation "
                          "was not specified thus it will be set to zero")

        ########################################################################################
        #
        #   Initialize Fourier transform for efficient calculations
        #
        ########################################################################################

        # Load FFTW wisdom if saved before
        try:
            with open('fftw_wisdom', 'rb') as f:
                pyfftw.import_wisdom(pickle.load(f))

                print("\nFFTW wisdom has been loaded\n")
        except IOError:
            pass

        # allocate the arrays for quantum classical hybrid wave function
        shape = (self.P_gridDIM, self.X_gridDIM)

        self.Upsilon1 = pyfftw.empty_aligned(shape, dtype=np.complex)
        self.Upsilon2 = pyfftw.empty_aligned(shape, dtype=np.complex)

        self.Upsilon1_copy = pyfftw.empty_aligned(shape, dtype=np.complex)
        self.Upsilon2_copy = pyfftw.empty_aligned(shape, dtype=np.complex)

        # Arrays to save the derivatives
        self.diff_Upsilon1_X = pyfftw.empty_aligned(shape, dtype=np.complex)
        self.diff_Upsilon1_P = pyfftw.empty_aligned(shape, dtype=np.complex)

        self.diff_Upsilon2_X = pyfftw.empty_aligned(shape, dtype=np.complex)
        self.diff_Upsilon2_P = pyfftw.empty_aligned(shape, dtype=np.complex)

        # allocate array for saving the classical density
        self.classical_rho = np.empty_like(self.Upsilon1)

        #  FFTW settings to achive good performace. For details see
        # https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html#pyfftw.FFTW
        fftw_params = dict(
            flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
            threads=ne.nthreads,
        )

        # p x -> theta x
        self.transform_p2theta =  pyfftw.FFTW(
            self.Upsilon1_copy, self.Upsilon1, axes=(0,), direction='FFTW_FORWARD', **fftw_params
        )

        # theta x  ->  p x
        self.transform_theta2p = pyfftw.FFTW(
            self.Upsilon1_copy, self.Upsilon1, axes=(0,), direction='FFTW_BACKWARD', **fftw_params
        )

        # p x  ->  p lambda
        self.transform_x2lambda = pyfftw.FFTW(
            self.Upsilon1_copy, self.Upsilon1, axes=(1,), direction='FFTW_FORWARD', **fftw_params
        )

        # p lambda  ->  p x
        self.transform_lambda2x = pyfftw.FFTW(
            self.Upsilon1_copy, self.Upsilon1, axes=(1,), direction='FFTW_BACKWARD', **fftw_params
        )

        # Save FFTW wisdom
        with open('fftw_wisdom', 'wb') as f:
            pickle.dump(pyfftw.export_wisdom(), f)

        ########################################################################################
        #
        #   Initialize grids
        #
        ########################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2.*self.X_amplitude / self.X_gridDIM
        self.dP = 2.*self.P_amplitude / self.P_gridDIM

        # pre-compute the volume element in phase space
        self.dXdP = self.dX * self.dP

        # The coordinate grid
        self.kX = np.arange(self.X_gridDIM)[np.newaxis, :]
        self.X = (self.kX - self.X_gridDIM / 2) * self.dX

        # The momentum grid
        self.kP = np.arange(self.P_gridDIM)[:, np.newaxis]
        self.P = (self.kP - self.P_gridDIM / 2) * self.dP

        # Lambda grid (variable conjugate to the coordinate)
        self.Lambda = (self.kX - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        # Theta grid (variable conjugate to the momentum)
        self.Theta = (self.kP - self.P_gridDIM / 2) * (np.pi / self.P_amplitude)

        ########################################################################################
        #
        #   Generate codes and allocate the space
        #
        ########################################################################################

        self.code_exp_diff_K = "exp(-0.5j * Lambda * dt * ({}))".format(self.diff_K)

        # allocate the array to pre-calculate the exponent
        self.exp_diff_K = np.empty_like(self.Upsilon1)

        ########################################################################################
        #
        #   Generate codes for components of the P operator (see the supplementary material)
        #       P = exp(1j * a * (c0 + c1 * sigma_1 + c2 * sigma_2 + c3 * sigma_3))
        #
        #
        ########################################################################################

        # Note that to avoid division by zero in the following four equations,  the ratio
        # 1 / {b} was modified to 1 / ({b} + 1e-100)

        P11 = "exp(1j * {{a}} * {{c0}}) * (cos({{a}} * {b}) + 1j * {{c3}} * sin({{a}} * {b}) / ({b} + 1e-100))".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P12 = "exp(1j * {{a}} * {{c0}}) * 1j * ({{c1}} - 1j * {{c2}}) * sin({{a}} * {b}) / ({b} + 1e-100)".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P21 = "exp(1j * {{a}} * {{c0}}) * 1j * ({{c1}} + 1j * {{c2}}) * sin({{a}} * {b}) / ({b} + 1e-100)".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        P22 = "exp(1j * {{a}} * {{c0}}) * (cos({{a}} * {b}) - 1j * {{c3}} * sin({{a}} * {b}) / ({b} + 1e-100))".format(
            b="sqrt({c1} ** 2 + {c2} ** 2 + {c3} ** 2)"
        )

        ################################## Generate code for Pc = exp(C) ##################################
        Pc_params = dict(
            a="(dt / 8.)",
            c0="( X * ({diff_U}) + P * ({diff_K}) - 2. * (({K}) + ({U})) )".format(
                diff_U=self.diff_U,
                diff_K=self.diff_K,
                K=self.K,
                U=self.U,
            ),
            c1="( X * ({}) - 2. * ({}) )".format(self.diff_f1, self.f1),
            c2="( X * ({}) - 2. * ({}) )".format(self.diff_f2, self.f2),
            c3="( X * ({}) - 2. * ({}) )".format(self.diff_f3, self.f3),
        )

        self.code_Pc_11 = P11.format(**Pc_params)
        self.code_Pc_12 = P12.format(**Pc_params)
        self.code_Pc_21 = P21.format(**Pc_params)
        self.code_Pc_22 = P22.format(**Pc_params)

        # Allocate arrays where the pre-calculated coefficients will be stored
        self.Pc_11 = np.empty_like(self.Upsilon1)
        self.Pc_12 = np.empty_like(self.Upsilon1)
        self.Pc_21 = np.empty_like(self.Upsilon1)
        self.Pc_22 = np.empty_like(self.Upsilon1)

        ################################## Generate code for Pb = exp(B) ##################################
        Pb_params = dict(
            a="(Theta * dt)",
            c0="(({}) + 1j * D * Theta)".format(self.diff_U),
            c1="({})".format(self.diff_f1),
            c2="({})".format(self.diff_f2),
            c3="({})".format(self.diff_f3),
        )

        self.code_expB_Upsilon1 = "({}) * Upsilon1_copy + ({}) * Upsilon2_copy".format(P11, P12).format(**Pb_params)
        self.code_expB_Upsilon2 = "({}) * Upsilon1_copy + ({}) * Upsilon2_copy".format(P21, P22).format(**Pb_params)

    def propagate(self, time_steps=1):
        """
        Time propagate the hybrid wavefunction saved in self.Upsilon1 and self.Upsilon2
        :param time_steps: number of self.dt time increments to make
        :return: self
        """
        # pseudonyms
        Upsilon1 = self.Upsilon1
        Upsilon2 = self.Upsilon2

        Upsilon1_copy = self.Upsilon1_copy
        Upsilon2_copy = self.Upsilon2_copy

        exp_diff_K = self.exp_diff_K

        for _ in range(time_steps):

            ############################################################################################
            #
            #   Single step propagation
            #
            ############################################################################################

            # make half a step in time
            self.t += 0.5 * self.dt

            # Pre calculate Pc = exp(C)
            ne.evaluate(self.code_Pc_11, local_dict=vars(self), out=self.Pc_11)
            ne.evaluate(self.code_Pc_12, local_dict=vars(self), out=self.Pc_12)
            ne.evaluate(self.code_Pc_21, local_dict=vars(self), out=self.Pc_21)
            ne.evaluate(self.code_Pc_22, local_dict=vars(self), out=self.Pc_22)

            # Apply exp(C)
            ne.evaluate(
                "(-1) ** kX * (Pc_11 * Upsilon1 + Pc_12 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon1_copy
            )
            ne.evaluate(
                "(-1) ** kX * (Pc_21 * Upsilon1 + Pc_22 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon2_copy
            )

            # p x  ->  p lambda
            self.transform_x2lambda(Upsilon1_copy, Upsilon1)
            self.transform_x2lambda(Upsilon2_copy, Upsilon2)

            # Pre-calculate exp(diff_K)
            ne.evaluate(self.code_exp_diff_K, local_dict=vars(self), out=exp_diff_K)

            # Apply exp(diff_K)

            #Upsilon1 *= exp_diff_K
            ne.evaluate("Upsilon1 * exp_diff_K", out=Upsilon1)
            #Upsilon2 *= exp_diff_K
            ne.evaluate("Upsilon2 * exp_diff_K", out=Upsilon2)

            # p lambda  ->  p x
            self.transform_lambda2x(Upsilon1, Upsilon1_copy)
            self.transform_lambda2x(Upsilon2, Upsilon2_copy)

            # Apply exp(C)
            ne.evaluate(
                "(-1) ** kP * (Pc_11 * Upsilon1_copy + Pc_12 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon1
            )
            ne.evaluate(
                "(-1) ** kP * (Pc_21 * Upsilon1_copy + Pc_22 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon2
            )

            # p x -> theta x
            self.transform_p2theta(Upsilon1, Upsilon1_copy)
            self.transform_p2theta(Upsilon2, Upsilon2_copy)

            # Apply exp(B)
            ne.evaluate(self.code_expB_Upsilon1, local_dict=vars(self), out=Upsilon1)
            ne.evaluate(self.code_expB_Upsilon2, local_dict=vars(self), out=Upsilon2)

            # theta x  ->  p x
            self.transform_theta2p(Upsilon1, Upsilon1_copy)
            self.transform_theta2p(Upsilon2, Upsilon2_copy)

            # Apply exp(C)
            ne.evaluate(
                "(-1) ** kP * (Pc_11 * Upsilon1_copy + Pc_12 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon1
            )
            ne.evaluate(
                "(-1) ** kP * (Pc_21 * Upsilon1_copy + Pc_22 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon2
            )

            # p x  ->  p lambda
            self.transform_x2lambda(Upsilon1, Upsilon1_copy)
            self.transform_x2lambda(Upsilon2, Upsilon2_copy)

            # Apply exp(diff_K)

            #Upsilon1_copy *= exp_diff_K
            ne.evaluate("Upsilon1_copy * exp_diff_K", out=Upsilon1_copy)
            #Upsilon2_copy *= exp_diff_K
            ne.evaluate("Upsilon2_copy * exp_diff_K", out=Upsilon2_copy)

            # p lambda  ->  p x
            self.transform_lambda2x(Upsilon1_copy, Upsilon1)
            self.transform_lambda2x(Upsilon2_copy, Upsilon2)

            # Apply exp(C)
            ne.evaluate(
                "(-1) ** kX * (Pc_11 * Upsilon1 + Pc_12 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon1_copy
            )
            ne.evaluate(
                "(-1) ** kX * (Pc_21 * Upsilon1 + Pc_22 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon2_copy
            )

            # Since the resulats of the previous step are saved in copies,
            # swap the references between the original and copies
            Upsilon1, Upsilon1_copy = Upsilon1_copy, Upsilon1
            Upsilon2, Upsilon2_copy = Upsilon2_copy, Upsilon2

            # synchronize pseudonyms with originals
            self.Upsilon1 = Upsilon1
            self.Upsilon2 = Upsilon2

            self.Upsilon1_copy = Upsilon1_copy
            self.Upsilon2_copy = Upsilon2_copy

            # make half a step in time
            self.t += 0.5 * self.dt

            # normalization
            self.normalize()

            # calculate the Ehrenfest theorems
            # self.get_Ehrenfest()

        return self

    def normalize(self):
        """
        Normalize the hybrid wave function
        :return: None
        """
        Upsilon1 = self.Upsilon1
        Upsilon2 = self.Upsilon2

        # get the normalization constant
        N = np.sqrt(
            ne.evaluate("sum(abs(Upsilon1) ** 2 + abs(Upsilon2) ** 2)").real * self.dXdP
        )

        #Upsilon1 /= N
        ne.evaluate("Upsilon1 / N", out=Upsilon1)

        #Upsilon2 /= N
        ne.evaluate("Upsilon2 / N", out=Upsilon2)

    def get_quantum_rho(self):
        """
        Calculate the quantum density matrix and save it in self.quantum_rho
        :return: self.quantum_rho (2x2 numpy.array)
        """
        Upsilon1 = self.Upsilon1
        Upsilon2 = self.Upsilon2

        rho12 = ne.evaluate("sum(Upsilon1 * conj(Upsilon2))")

        self.quantum_rho = np.array([
            [ne.evaluate("sum(abs(Upsilon1) ** 2)"), rho12],
            [rho12.conj(), ne.evaluate("sum(abs(Upsilon2) ** 2)")]
        ])

        self.quantum_rho /= self.quantum_rho.trace()

        return self.quantum_rho

    def quantum_average(self, quantum_observable):
        """
        Calculate the expectation value of the observable acting only on the quantum degree of freedom
        :param quantum_observable: 2x2 numpy.array
        :return: float
        """
        return self.quantum_rho.dot(quantum_observable).trace().real

    def classical_average(self, classical_observable):
        """
        Calculate the expectation value of the observable acting only on the classical degree of freedom
        :param classical_observable: string to be evaluated by numexpr
        :return: float
        """
        return ne.evaluate(
            "sum(({}) * (abs(Upsilon1) ** 2 + abs(Upsilon2) ** 2))".format(classical_observable),
            local_dict=vars(self),
        ) * self.dXdP

    def get_Upsilon_gradinet(self):
        """
        Save the gradients of Upsilon1 and Upsilon2 into
        self.diff_Upsilon1_X, self.diff_Upsilon1_P, self.diff_Upsilon2_X, and self.diff_Upsilon2_P respectively
        :return: None
        """
        Upsilon1_copy = self.Upsilon1_copy
        Upsilon2_copy = self.Upsilon2_copy

        ############################################################################################
        #
        #   X derivatives
        #
        ############################################################################################

        ne.evaluate("(-1) ** kX * Upsilon1", local_dict=vars(self), out=Upsilon1_copy)
        ne.evaluate("(-1) ** kX * Upsilon2", local_dict=vars(self), out=Upsilon2_copy)

        # p x  ->  p lambda
        self.transform_x2lambda(Upsilon1_copy, self.diff_Upsilon1_X)
        self.transform_x2lambda(Upsilon2_copy, self.diff_Upsilon2_X)

        ne.evaluate("diff_Upsilon1_X * 1j * Lambda", local_dict=vars(self), out=Upsilon1_copy)
        ne.evaluate("diff_Upsilon2_X * 1j * Lambda", local_dict=vars(self), out=Upsilon2_copy)

        # p lambda  ->  p x
        self.transform_lambda2x(Upsilon1_copy, self.diff_Upsilon1_X)
        self.transform_lambda2x(Upsilon2_copy, self.diff_Upsilon2_X)

        ne.evaluate("(-1) ** kX * diff_Upsilon1_X", local_dict=vars(self), out=self.diff_Upsilon1_X)
        ne.evaluate("(-1) ** kX * diff_Upsilon2_X", local_dict=vars(self), out=self.diff_Upsilon2_X)

        ############################################################################################
        #
        #   P derivatives
        #
        ############################################################################################

        ne.evaluate("(-1) ** kP * Upsilon1", local_dict=vars(self), out=Upsilon1_copy)
        ne.evaluate("(-1) ** kP * Upsilon2", local_dict=vars(self), out=Upsilon2_copy)

        # p x -> theta x
        self.transform_p2theta(Upsilon1_copy, self.diff_Upsilon1_P)
        self.transform_p2theta(Upsilon2_copy, self.diff_Upsilon2_P)

        ne.evaluate("diff_Upsilon1_P * 1j * Theta", local_dict=vars(self), out=Upsilon1_copy)
        ne.evaluate("diff_Upsilon2_P * 1j * Theta", local_dict=vars(self), out=Upsilon2_copy)

        # theta x -> p x
        self.transform_theta2p(Upsilon1_copy, self.diff_Upsilon1_P)
        self.transform_theta2p(Upsilon2_copy, self.diff_Upsilon2_P)

        ne.evaluate("(-1) ** kP * diff_Upsilon1_P", local_dict=vars(self), out=self.diff_Upsilon1_P)
        ne.evaluate("(-1) ** kP * diff_Upsilon2_P", local_dict=vars(self), out=self.diff_Upsilon2_P)

    def get_classical_rho(self):
        """
        Calculate the Liouville density and save it in self.classical_rho
        :return: self.classical_rho.real
        """
        self.get_Upsilon_gradinet()

        ne.evaluate(
            "2. * abs(Upsilon1) ** 2 + 2. * abs(Upsilon2) ** 2 + real("\
            "   X * (Upsilon1 * conj(diff_Upsilon1_X) + Upsilon2 * conj(diff_Upsilon2_X)) + "\
            "   P * (Upsilon1 * conj(diff_Upsilon1_P) + Upsilon2 * conj(diff_Upsilon2_P)) + "\
            "   2.j * (diff_Upsilon1_X * conj(diff_Upsilon1_P) + diff_Upsilon2_X * conj(diff_Upsilon2_P))"
            ")",
            local_dict = vars(self),
            out=self.classical_rho
        )

        return self.classical_rho.real

    def set_wavefunction(self, new_Upsilon1, new_Upsilon2):
        """
        Set the initial Wigner function
        :param new_Upsilon1, new_Upsilon2: a 2D numpy array or sting containing the components of the function
        :return: self
        """
        if isinstance(new_Upsilon1, str):
            ne.evaluate(new_Upsilon1, local_dict=vars(self), out=self.Upsilon1)

        elif isinstance(new_Upsilon1, np.ndarray):
            np.copyto(self.Upsilon1, new_Upsilon1)

        else:
            raise ValueError("new_Upsilon1 must be either string or numpy.array")

        if isinstance(new_Upsilon2, str):
            ne.evaluate(new_Upsilon2, local_dict=vars(self), out=self.Upsilon2)

        elif isinstance(new_Upsilon2, np.ndarray):
            np.copyto(self.Upsilon2, new_Upsilon2)

        else:
            raise ValueError("new_Upsilon2 must be either string or numpy.array")

        self.normalize()

        return self


##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(QCHybrid.__doc__)

    class VisualizeHybrid:
        """
        Class to visualize the Wigner function function dynamics in phase space.
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
                norm=WignerNormalize(vmin=-0.01, vmax=0.1)
                #norm=WignerSymLogNorm(linthresh=1e-16, vmin=-0.01, vmax=0.1)
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

            # self.fig.colorbar(self.img_Upsilon1)

        def set_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.quant_sys = QCHybrid(
                t=0,

                dt=0.05,

                X_gridDIM=256,
                X_amplitude=10.,

                P_gridDIM=256,
                P_amplitude=10.,

                K="0.5 * P ** 2",
                diff_K="P",

                #omega=np.random.uniform(1, 3),
                omega=1.,

                U="0.5 * (omega * X) ** 2",
                diff_U="omega ** 2 * X",

                alpha=0.1,

                f1="alpha * X + alpha",
                diff_f1="alpha",

                f2="alpha * X + alpha",
                diff_f2="alpha",

                f3="alpha * X + alpha",
                diff_f3="alpha",
            )

            Upsilon = "exp( -{sigma} * (X - {X0}) ** 2 - (1. / {sigma}) * (P - {P0}) ** 2 )".format(
                sigma=np.random.uniform(1., 2.),
                P0=np.random.uniform(-3., 3.),
                X0=np.random.uniform(-3., 3.),
            )

            # set randomised initial condition
            self.quant_sys.set_wavefunction(Upsilon, "0. * X + 0. * P")

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """



            #print(np.linalg.eigvalsh(self.quant_sys.quantum_rho))

            #print(quantum_rho.dot(quantum_rho).trace())

            # propagate the wigner function
            self.img_Upsilon1.set_array(
                self.quant_sys.get_classical_rho()
            )

            self.img_Upsilon2.set_array(
                self.quant_sys.Upsilon1.imag
            )

            #avX1 = ne.evaluate("sum(classical_rho * P)", local_dict=vars(self.quant_sys)) * self.quant_sys.dXdP
            #avX2 = self.quant_sys.classical_average("P")

            #print(avX1 / avX2)

            #print(self.quant_sys.classical_average("1"))
            #print(self.quant_sys.classical_rho.sum() * self.quant_sys.dXdP)

            self.quant_sys.propagate(10)

            return self.img_Upsilon1, self.img_Upsilon2


    fig = plt.gcf()
    visualizer = VisualizeHybrid(fig)
    animation = FuncAnimation(
        fig, visualizer, frames=np.arange(100), repeat=True, blit=True
    )
    plt.show()