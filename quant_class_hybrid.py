import numpy as np
import pyfftw
import pickle
import warnings
from types import MethodType, FunctionType
import numexpr as ne
from numexpr import evaluate


class QCHybrid(object):
    """
    The Koopman-von Neumann formulation of hybrid classical-quantum systems
        https://arxiv.org/abs/1802.04787
    """

    def __init__(self, *, X_gridDIM, X_amplitude, P_gridDIM, P_amplitude, U, diff_U, K, diff_K,
                 f1, diff_f1, f2, diff_f2, f3, diff_f3, dt, t, D=0, **kwargs):
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
        :param D: (optional) the diffusion coefficient to mitigate the velocity filamentation
        :param kwargs: other parameters
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
        self.D = D

        # save all the other attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            # otherwise bind it as a property
            else:
                setattr(self, name, value)

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

        # allocate arrays for the components of the hybrid density matrix
        self.D11 = np.empty_like(self.Upsilon1)
        self.D12 = np.empty_like(self.Upsilon1)
        self.D22 = np.empty_like(self.Upsilon1)

        ########################################################################################

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
        #   Generate grids for self.get_Upsilon_gradient
        #
        ########################################################################################

        from scipy.sparse import diags

        # generate the matrix corresponding to the second order finite difference operator
        fdiff = diags([-1., 0, 1.], [-1, 0, 1], shape=self.Upsilon1.shape, dtype=self.Upsilon1.dtype, format='csr')
        fdiff[0, -1] = -1.
        fdiff[-1, 0] = 1.

        # Diagonalize the matrix using FFT

        # Convert the matrix to the dense array
        fdiff.toarray(out=self.Upsilon1)

        self.transform_x2lambda(self.Upsilon1, self.Upsilon1_copy)
        self.transform_theta2p(self.Upsilon1_copy, self.Upsilon1)

        eigenvals = self.Upsilon1.diagonal().imag

        # Modified lambda grid (variable conjugate to the coordinate)
        self.Lambda_ = eigenvals / (2. * self.dX)
        self.Lambda_ = self.Lambda_[np.newaxis, :]

        # Modified theta grid (variable conjugate to the momentum)
        self.Theta_ = eigenvals / (2. * self.dP)
        self.Theta_ = self.Theta_[:, np.newaxis]

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

        ################################## Generate code for Pmh = exp(-H) ################################

        # parameters of the hamiltonian
        h_params = dict(
            c0="({} + {})".format(self.K, self.U),
            c1="({})".format(self.f1),
            c2="({})".format(self.f2),
            c3="({})".format(self.f3),
        )

        self.code_exp_minus_H1 = "({}) * Upsilon1 + ({}) * Upsilon2".format(P11, P12).format(a=1.j, **h_params)
        self.code_exp_minus_H2 = "({}) * Upsilon1 + ({}) * Upsilon2".format(P21, P22).format(a=1.j, **h_params)

        ################################## Generate code for Pph = exp(+H) ################################

        self.code_exp_plus_H1 = "where(real(abs(Upsilon1)) < 1e-9, 0, ({}) * Upsilon1) + where(real(abs(Upsilon2)) < 1e-9, 0, ({}) * Upsilon2)".format(P11, P12).format(a=-1.j, **h_params)
        self.code_exp_plus_H2 = "where(real(abs(Upsilon1)) < 1e-9, 0, ({}) * Upsilon1) + where(real(abs(Upsilon2)) < 1e-9, 0, ({}) * Upsilon2)".format(P21, P22).format(a=-1.j, **h_params)
        #self.code_exp_plus_H2 = "({}) * Upsilon1 + ({}) * Upsilon2".format(P21, P22).format(a=-1.j, **h_params)

        ########################################################################################

        # Call the initialization procedure
        self.post_initialization()

    def post_initialization(self):
        """
        Place holder for the user defined function to be call after at the constructor
        """
        pass

    def propagate(self, time_steps=1):
        """
        Time propagate the hybrid wavefunction saved in self.Upsilon1 and self.Upsilon2
        :param time_steps: number of self.dt time increments to make
        :return: self
        """
        for _ in range(time_steps):

            ############################################################################################
            #
            #   Single step propagation
            #
            ############################################################################################
            # pseudonyms
            Upsilon1 = self.Upsilon1
            Upsilon2 = self.Upsilon2

            Upsilon1_copy = self.Upsilon1_copy
            Upsilon2_copy = self.Upsilon2_copy

            exp_diff_K = self.exp_diff_K

            # make half a step in time
            self.t += 0.5 * self.dt

            # Pre calculate Pc = exp(C)
            evaluate(self.code_Pc_11, local_dict=vars(self), out=self.Pc_11)
            evaluate(self.code_Pc_12, local_dict=vars(self), out=self.Pc_12)
            evaluate(self.code_Pc_21, local_dict=vars(self), out=self.Pc_21)
            evaluate(self.code_Pc_22, local_dict=vars(self), out=self.Pc_22)

            # Apply exp(C)
            evaluate(
                "(-1) ** kX * (Pc_11 * Upsilon1 + Pc_12 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon1_copy
            )
            evaluate(
                "(-1) ** kX * (Pc_21 * Upsilon1 + Pc_22 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon2_copy
            )

            # p x  ->  p lambda
            self.transform_x2lambda(Upsilon1_copy, Upsilon1)
            self.transform_x2lambda(Upsilon2_copy, Upsilon2)

            # Pre-calculate exp(diff_K)
            evaluate(self.code_exp_diff_K, local_dict=vars(self), out=exp_diff_K)

            # Apply exp(diff_K)

            #Upsilon1 *= exp_diff_K
            evaluate("Upsilon1 * exp_diff_K", out=Upsilon1)
            #Upsilon2 *= exp_diff_K
            evaluate("Upsilon2 * exp_diff_K", out=Upsilon2)

            # p lambda  ->  p x
            self.transform_lambda2x(Upsilon1, Upsilon1_copy)
            self.transform_lambda2x(Upsilon2, Upsilon2_copy)

            # Apply exp(C)
            evaluate(
                "(-1) ** kP * (Pc_11 * Upsilon1_copy + Pc_12 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon1
            )
            evaluate(
                "(-1) ** kP * (Pc_21 * Upsilon1_copy + Pc_22 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon2
            )

            # p x -> theta x
            self.transform_p2theta(Upsilon1, Upsilon1_copy)
            self.transform_p2theta(Upsilon2, Upsilon2_copy)

            # Apply exp(B)
            evaluate(self.code_expB_Upsilon1, local_dict=vars(self), out=Upsilon1)
            evaluate(self.code_expB_Upsilon2, local_dict=vars(self), out=Upsilon2)

            # theta x  ->  p x
            self.transform_theta2p(Upsilon1, Upsilon1_copy)
            self.transform_theta2p(Upsilon2, Upsilon2_copy)

            # Apply exp(C)
            evaluate(
                "(-1) ** kP * (Pc_11 * Upsilon1_copy + Pc_12 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon1
            )
            evaluate(
                "(-1) ** kP * (Pc_21 * Upsilon1_copy + Pc_22 * Upsilon2_copy)",
                local_dict=vars(self),
                out=Upsilon2
            )

            # p x  ->  p lambda
            self.transform_x2lambda(Upsilon1, Upsilon1_copy)
            self.transform_x2lambda(Upsilon2, Upsilon2_copy)

            # Apply exp(diff_K)

            #Upsilon1_copy *= exp_diff_K
            evaluate("Upsilon1_copy * exp_diff_K", out=Upsilon1_copy)
            #Upsilon2_copy *= exp_diff_K
            evaluate("Upsilon2_copy * exp_diff_K", out=Upsilon2_copy)

            # p lambda  ->  p x
            self.transform_lambda2x(Upsilon1_copy, Upsilon1)
            self.transform_lambda2x(Upsilon2_copy, Upsilon2)

            # Apply exp(C)
            evaluate(
                "(-1) ** kX * (Pc_11 * Upsilon1 + Pc_12 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon1_copy
            )
            evaluate(
                "(-1) ** kX * (Pc_21 * Upsilon1 + Pc_22 * Upsilon2)",
                local_dict=vars(self),
                out=Upsilon2_copy
            )

            # Since the resulats of the previous step are saved in copies,
            # swap the references between the original and copies
            self.Upsilon1, self.Upsilon1_copy = self.Upsilon1_copy, self.Upsilon1
            self.Upsilon2, self.Upsilon2_copy = self.Upsilon2_copy, self.Upsilon2

            # make half a step in time
            self.t += 0.5 * self.dt

            # normalization
            self.normalize()

            # call user defined post-processing
            self.post_processing()

        return self

    def post_processing(self):
        """
        Place holder for the user defined function to be call at the end of each propagation step
        """
        pass

    def normalize(self):
        """
        Normalize the hybrid wave function
        :return: None
        """
        Upsilon1 = self.Upsilon1
        Upsilon2 = self.Upsilon2

        # get the normalization constant
        N = np.sqrt(
            evaluate("sum(abs(Upsilon1) ** 2 + abs(Upsilon2) ** 2)").real * self.dXdP
        )

        #Upsilon1 /= N
        evaluate("Upsilon1 / N", out=Upsilon1)

        #Upsilon2 /= N
        evaluate("Upsilon2 / N", out=Upsilon2)

    def get_quantum_rho(self):
        """
        Calculate the quantum density matrix and save it in self.quantum_rho
        :return: self.quantum_rho (2x2 numpy.array)
        """
        Upsilon1 = self.Upsilon1
        Upsilon2 = self.Upsilon2

        rho12 = evaluate("sum(Upsilon1 * conj(Upsilon2))")

        self.quantum_rho = np.array([
            [evaluate("sum(abs(Upsilon1) ** 2)"), rho12],
            [rho12.conj(), evaluate("sum(abs(Upsilon2) ** 2)")]
        ])

        self.quantum_rho /= self.quantum_rho.trace()

        return self.quantum_rho

    def quantum_average(self, quantum_observable, calculate=True):
        """
        Calculate the expectation value of the observable acting only on the quantum degree of freedom
        :param quantum_observable: 2x2 numpy.array
        :param calculate: a boolean flag indicating whether the quantum rho is to be calculated
        :return: float
        """
        if calculate:
            self.get_quantum_rho()

        return self.quantum_rho.dot(quantum_observable).trace().real

    def classical_average(self, classical_observable, calculate=True):
        """
        Calculate the expectation value of the observable acting only on the classical degree of freedom
        :param classical_observable: string to be evaluated by numexpr
        :param calculate: a boolean flag indicating whether the classical rho is to be calculated
        :return: float
        """
        if calculate:
            self.get_classical_rho()

        return evaluate(
            "sum(({}) * classical_rho)".format(classical_observable),
            local_dict=vars(self),
        ) * self.dXdP

    def get_Upsilon_gradient(self):
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

        evaluate("(-1) ** kX * Upsilon1", local_dict=vars(self), out=Upsilon1_copy)
        evaluate("(-1) ** kX * Upsilon2", local_dict=vars(self), out=Upsilon2_copy)

        # p x  ->  p lambda
        self.transform_x2lambda(Upsilon1_copy, self.diff_Upsilon1_X)
        self.transform_x2lambda(Upsilon2_copy, self.diff_Upsilon2_X)

        evaluate("diff_Upsilon1_X * 1j * Lambda_", local_dict=vars(self), out=Upsilon1_copy)
        evaluate("diff_Upsilon2_X * 1j * Lambda_", local_dict=vars(self), out=Upsilon2_copy)

        # p lambda  ->  p x
        self.transform_lambda2x(Upsilon1_copy, self.diff_Upsilon1_X)
        self.transform_lambda2x(Upsilon2_copy, self.diff_Upsilon2_X)

        evaluate("(-1) ** kX * diff_Upsilon1_X", local_dict=vars(self), out=self.diff_Upsilon1_X)
        evaluate("(-1) ** kX * diff_Upsilon2_X", local_dict=vars(self), out=self.diff_Upsilon2_X)

        ############################################################################################
        #
        #   P derivatives
        #
        ############################################################################################

        evaluate("(-1) ** kP * Upsilon1", local_dict=vars(self), out=Upsilon1_copy)
        evaluate("(-1) ** kP * Upsilon2", local_dict=vars(self), out=Upsilon2_copy)

        # p x -> theta x
        self.transform_p2theta(Upsilon1_copy, self.diff_Upsilon1_P)
        self.transform_p2theta(Upsilon2_copy, self.diff_Upsilon2_P)

        evaluate("diff_Upsilon1_P * 1j * Theta_", local_dict=vars(self), out=Upsilon1_copy)
        evaluate("diff_Upsilon2_P * 1j * Theta_", local_dict=vars(self), out=Upsilon2_copy)

        # theta x -> p x
        self.transform_theta2p(Upsilon1_copy, self.diff_Upsilon1_P)
        self.transform_theta2p(Upsilon2_copy, self.diff_Upsilon2_P)

        evaluate("(-1) ** kP * diff_Upsilon1_P", local_dict=vars(self), out=self.diff_Upsilon1_P)
        evaluate("(-1) ** kP * diff_Upsilon2_P", local_dict=vars(self), out=self.diff_Upsilon2_P)

    def get_classical_rho(self, calculate=True):
        """
        Calculate the Liouville density and save it in self.classical_rho
        :param calculate: a boolean flag indicating whether the derivatives of Upsion are to be calculated
        :return: self.classical_rho.real
        """
        if calculate:
            self.get_Upsilon_gradient()

        evaluate(
            "2. * abs(Upsilon1) ** 2 + 2. * abs(Upsilon2) ** 2 + real("
            "   X * (Upsilon1 * conj(diff_Upsilon1_X) + Upsilon2 * conj(diff_Upsilon2_X)) + "
            "   P * (Upsilon1 * conj(diff_Upsilon1_P) + Upsilon2 * conj(diff_Upsilon2_P)) + "
            "   2.j * (diff_Upsilon1_X * conj(diff_Upsilon1_P) + diff_Upsilon2_X * conj(diff_Upsilon2_P))"
            ")",
            local_dict=vars(self),
            out=self.classical_rho
        )

        return self.classical_rho.real

    def get_hybrid_D(self, calculate=True):
        """
        Calculate the hybrid density matrix and save its components in self.D11, self.D12, and self.D22
        :param calculate: a boolean flag indicating whether the derivatives of Upsion are to be calculated
        :return: None
        """
        if calculate:
            self.get_Upsilon_gradient()

        evaluate(
            "2. * abs(Upsilon1) ** 2  + real( "
            "   X * Upsilon1 * conj(diff_Upsilon1_X) + P * Upsilon1 * conj(diff_Upsilon1_P) + "
            "   2.j * diff_Upsilon1_X * conj(diff_Upsilon1_P)"
            ")",
            local_dict=vars(self),
            out=self.D11
        )

        evaluate(
            "2. * Upsilon1 * conj(Upsilon2) + "
            "1j * (diff_Upsilon1_X * conj(diff_Upsilon2_P) - conj(diff_Upsilon2_X) * diff_Upsilon1_P) +"
            "0.5 * Upsilon1 * (X * conj(diff_Upsilon2_X) + P * conj(diff_Upsilon2_P)) + "
            "0.5 * conj(Upsilon2) * (X * diff_Upsilon1_X + P * diff_Upsilon1_P)",
            local_dict=vars(self),
            out=self.D12
        )

        evaluate(
            "2. * abs(Upsilon2) ** 2  + real( "
            "   X * Upsilon2 * conj(diff_Upsilon2_X) + P * Upsilon2 * conj(diff_Upsilon2_P) + "
            "   2.j * diff_Upsilon2_X * conj(diff_Upsilon2_P)"
            ")",
            local_dict=vars(self),
            out=self.D22
        )

    def hybrid_average(self, observable, calculate=True):
        """
        Obtain the expectation value of a general observable
        (which acts on both the classical and quantum degrees of freedom)
        :param observable: a list of strings to be evaluated by numexpr.
            observable[k] describes the k-th coefficient in front of the k-th Pauli matrix.
                observable = sum_{k=0}^3 \sigma_k observable[k]
        :param calculate:  a boolean flag indicating whether the hybrid density matrix needs to be evaluated
        :return: complex
        """
        assert len(observable) == 4, "observable should be a list of four strings to be evaluated by numexpr"

        if calculate:
            self.get_hybrid_D()

        return evaluate(
            "sum(({}) * (D11 + D22) + 2. * real(D12) * ({}) - 2. * imag(D12) * ({}) + (D11 - D22) * ({}))".format(
                *observable
            ),
            local_dict=vars(self)
        ) * self.dXdP

    def gaussian_filter(self, Upsilon, sigma_x, sigma_p):
        """
        Convolve Upsilon with a Gaussian. The result will be stored in Upslion
        :param Upsilon: numpy.array of the same shape as self.Upsilon1
        :param sigma_x: the standard deviation in units of self.dX
        :param sigma_p: the standard deviation in units of self.dP
        :return: None
        """
        Upsilon_copy = self.Upsilon1_copy

        evaluate("(-1) ** (kX + kP) * Upsilon", global_dict=vars(self), out=Upsilon_copy)

        # p x  ->  p lambda
        self.transform_x2lambda(Upsilon_copy, Upsilon)

        # p lambda -> theta lambda
        self.transform_p2theta(Upsilon, Upsilon_copy)

        alpha_theta = 0.25 * (sigma_p * self.dP) ** 2
        alpha_lambda = 0.25 * (sigma_x * self.dX) ** 2

        evaluate(
            "Upsilon_copy * exp(-alpha_theta * Theta ** 2 -alpha_lambda * Lambda ** 2)",
            global_dict=vars(self),
            out=Upsilon_copy
        )

        # theta lambda -> p lambda
        self.transform_theta2p(Upsilon_copy, Upsilon)

        # p lambda  ->  p x
        self.transform_lambda2x(Upsilon, Upsilon_copy)

        evaluate("(-1) ** (kX + kP) * Upsilon_copy", global_dict=vars(self), out=Upsilon)

    def rotate(self, Upsilon, alpha):
        """
        Rotate the array Upsilon by angle alpha. Results will be saved in Upsilon
        :param Upsilon: numpy.array
        :param alpha: float
        :return: None
        """
        Upsilon1_copy = self.Upsilon1_copy

        evaluate("(-1) ** (kX + kP) * Upsilon", global_dict=vars(self), out=Upsilon)

        # Shear X
        self.transform_x2lambda(Upsilon, Upsilon1_copy)
        evaluate("exp(-1j * tan(0.5 * alpha) * P * Lambda) * Upsilon1_copy", out=Upsilon1_copy, global_dict=vars(self))
        self.transform_lambda2x(Upsilon1_copy, Upsilon)

        # Shear Y
        self.transform_p2theta(Upsilon, Upsilon1_copy)
        evaluate("exp(1j * sin(alpha) * Theta * X) * Upsilon1_copy", out=Upsilon1_copy, global_dict=vars(self))
        self.transform_theta2p(Upsilon1_copy, Upsilon)

        # Shear X
        self.transform_x2lambda(Upsilon, Upsilon1_copy)
        evaluate("exp(-1j * tan(0.5 * alpha) * P * Lambda) * Upsilon1_copy", out=Upsilon1_copy, global_dict=vars(self))
        self.transform_lambda2x(Upsilon1_copy, Upsilon)

        evaluate("(-1) ** (kX + kP) * Upsilon", global_dict=vars(self), out=Upsilon)

    def exp_minus_H_Upsilon(self):
        """
        Calculate exp(-H) Upsilon and save the result in self.Upsilon
        :return: None
        """
        evaluate(self.code_exp_minus_H1, local_dict=vars(self), out=self.Upsilon1_copy)
        evaluate(self.code_exp_minus_H2, local_dict=vars(self), out=self.Upsilon2_copy)

        self.Upsilon1, self.Upsilon1_copy = self.Upsilon1_copy, self.Upsilon1
        self.Upsilon2, self.Upsilon2_copy = self.Upsilon2_copy, self.Upsilon2

    def exp_plus_H_Upsilon(self):
        """
        Calculate exp(+H) Upsilon and save the result in self.Upsilon
        :return: None
        """
        evaluate(self.code_exp_plus_H1, local_dict=vars(self), out=self.Upsilon1_copy)
        evaluate(self.code_exp_plus_H2, local_dict=vars(self), out=self.Upsilon2_copy)

        self.Upsilon1, self.Upsilon1_copy = self.Upsilon1_copy, self.Upsilon1
        self.Upsilon2, self.Upsilon2_copy = self.Upsilon2_copy, self.Upsilon2

    def set_wavefunction(self, new_Upsilon1, new_Upsilon2):
        """
        Set the initial Wigner function
        :param new_Upsilon1, new_Upsilon2: a 2D numpy array or sting containing the components of the function
        :return: self
        """
        if isinstance(new_Upsilon1, str):
            evaluate(new_Upsilon1, local_dict=vars(self), out=self.Upsilon1)

        elif isinstance(new_Upsilon1, np.ndarray):
            np.copyto(self.Upsilon1, new_Upsilon1)

        else:
            raise ValueError("new_Upsilon1 must be either string or numpy.array")

        if isinstance(new_Upsilon2, str):
            evaluate(new_Upsilon2, local_dict=vars(self), out=self.Upsilon2)

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
                dt=0.05,

                X_gridDIM=256,
                X_amplitude=10.,

                P_gridDIM=256,
                P_amplitude=10.,

                K="0.5 * P ** 2",
                diff_K="P",

                #omega=np.random.uniform(1, 3),
                omega=1,

                U="0.5 * (omega * X) ** 2",
                diff_U="omega ** 2 * X",

                alpha=1,

                f1="alpha * X",
                diff_f1="alpha",

                f2="alpha * X",
                diff_f2="alpha",

                f3="alpha * X",
                diff_f3="alpha",

                # add facility to calculate the expectation value of energy
                post_initialization=post_initialization,
                post_processing=post_processing,
            )

            Upsilon = "exp( -{sigma} * (X - {X0}) ** 2 - (1. / {sigma}) * (P - {P0}) ** 2 )".format(
                 sigma=np.random.uniform(1., 2.),
                 P0=np.random.uniform(-3., 3.),
                 X0=np.random.uniform(-3., 3.),
            )

            # set randomised initial condition
            self.quant_sys.set_wavefunction(Upsilon, Upsilon)

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            quant_sys = self.quant_sys

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

            print(
                quant_sys.classical_average("X", False) - quant_sys.hybrid_average(("X", "0.", "0.", "0."), False)
            )

            sigma_x = ((1, 0), (0, -1))

            print(
                quant_sys.quantum_average(sigma_x) - quant_sys.hybrid_average(("0", "0", "0", "1"), False)

            )

            print("\n")

            self.img_Upsilon2.set_array(
                quant_sys.Upsilon1.real
            )

            #avX1 = evaluate("sum(classical_rho * P)", local_dict=vars(self.quant_sys)) * self.quant_sys.dXdP
            #avX2 = self.quant_sys.classical_average("P")

            #print(avX1 / avX2)

            #print(self.quant_sys.classical_average("1"))
            #print(self.quant_sys.classical_rho.sum() * self.quant_sys.dXdP)

            self.quant_sys.propagate(10)

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
