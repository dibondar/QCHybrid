import numpy as np
from numexpr import evaluate
from scipy.fftpack import dctn
from scipy.linalg import expm
from scipy.sparse import csc_matrix, hstack, bmat

########################################################################################################################
#
#   miscellaneous functions used in the class below
#
########################################################################################################################


def get_chebyshev_list(x: np.array, chebyshev0: np.array, chebyshev1: np.array, n_basis_vect:int) -> list:
    """
    Depending on the value of chebyshev0 and chebyshve1, get [T_n(x) for n in range(n_basis_vect)] or
    [U_n(x) for n in range(n_basis_vect)] using the recurrence relations for the Chebyshev polynomials
    :param x: np.array
    :param n_basis_vect: int
    :return: list
    """
    chebyshev_n_minus_1 = chebyshev0.copy()
    chebyshev_n = chebyshev1.copy()

    chebyshev_list = [chebyshev_n_minus_1.copy(), chebyshev_n.copy()]

    chebyshev_n_plus_1 = np.zeros_like(x)

    for _ in range(n_basis_vect - 2):
        evaluate("2. * x * chebyshev_n - chebyshev_n_minus_1", out=chebyshev_n_plus_1)

        chebyshev_list.append(chebyshev_n_plus_1.copy())

        chebyshev_n_plus_1, chebyshev_n_minus_1, chebyshev_n = chebyshev_n_minus_1, chebyshev_n, chebyshev_n_plus_1

    return chebyshev_list

########################################################################################################################
#
#   main class declaration
#
########################################################################################################################

class CHybridProp(object):
    """
    The numerical propagator for the hybrid equation of motion using the Chebyshev polynomials as basis.

        upsilon1 = \sum_{n,m=1}^{n_basis_vect - 1} upsilon1_coeff[n, m] T[n](q)T[m](p)

    """
    def __init__(self, *, n_basis_vect: int = 20, dt: float = 0.01,
                 h: str = "0. * p + 0. * q", diff_q_h: str = "0. * p + 0. * q",
                 diff_p_h: str = "0. * p + 0. * q", f1: str = "0. * q", f2: str = "0. * q", f3: str = "0. * q",
                 diff_q_f1: str = "0. * q", diff_q_f2: str = "0. * q", diff_q_f3: str = "0. * q",):
        """
        :param n_basis_vect: the number of the Chebyshev polynomials to use
        :param dt: the time increment
        :param h: the hamiltonian - a string to be evaluated by numexpr representing a function of q and p
        :param diff_q_h: the coordinate derivative of the hamiltonian - a string to be evaluated by numexpr
        :param diff_p_h: the momentum derivative of the hamiltonian - a string to be evaluated by numexpr
        :param f1: the quantum classical coupling - a string to be evaluated by numexpr
        :param diff_q_f1: the derivative of the quantum classical coupling - a string to be evaluated by numexpr
        :param f2: the quantum classical coupling - a string to be evaluated by numexpr
        :param diff_q_f2: the derivative of the quantum classical coupling - a string to be evaluated by numexpr
        :param f3: the quantum classical coupling - a string to be evaluated by numexpr
        :param diff_q_f3: the derivative of the quantum classical coupling - a string to be evaluated by numexpr
        """

        # Saving parameters
        self.n_basis_vect = n_basis_vect
        self.dt = dt

        self.h = h
        self.diff_q_h = diff_q_h
        self.diff_p_h = diff_p_h

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.diff_q_f1 = diff_q_f1
        self.diff_q_f2 = diff_q_f2
        self.diff_q_f3 = diff_q_f3

        ################################################################################################################

        # allocate the array of coefficients for hybrid wavefunctions

        # make sure that both components of the wavefunction lye in one continuous memory block
        # this is convenient for self.propagate
        self._flatten_upsilon = np.zeros(2 * self.n_basis_vect ** 2, dtype=np.complex)

        self.upsilon1_coeff = self._flatten_upsilon[:self.n_basis_vect**2].reshape(self.n_basis_vect, self.n_basis_vect)
        self.upsilon2_coeff = self._flatten_upsilon[self.n_basis_vect**2:].reshape(self.n_basis_vect, self.n_basis_vect)

        # a copy used in self.propagate
        self._flatten_upsilon_copy = self._flatten_upsilon.copy()

        ################################################################################################################

        # variables for pre-calculating Chebyshev polynomials
        self._p = None
        self._q = None

        # generate grid points for computing matrix elements and expansion
        chebyshev_zeros = np.cos(np.pi * (np.arange(self.n_basis_vect) + 0.5) / self.n_basis_vect)
        self.comp_p = chebyshev_zeros[np.newaxis, :]
        self.comp_q = chebyshev_zeros[:, np.newaxis]

        # save the liouvillian matrix
        self.get_liouvillian_matrix()
        # self.propagator = expm(-self.dt * 1j * self.liouvillian)

    def get_liouvillian_matrix(self):
        """
        :return:
        """
        p = self.comp_p
        q = self.comp_q

        self.save_chebyshev(q, p)

        # save separately the derivatives of the CHebyshev t polynomials
        diff_chebyshev_t_p = [0.,] + [(n + 1) * Up for n, Up in enumerate(self.chebyshev_u_p[:-1])]
        diff_chebyshev_t_q = [0.,] + [(n + 1) * Uq for n, Uq in enumerate(self.chebyshev_u_q[:-1])]

        # evaluate different part of the Liouvillian
        h = evaluate(self.h)
        diff_q_h = evaluate(self.diff_q_h)
        diff_p_h = evaluate(self.diff_p_h)

        f1 = evaluate(self.f1)
        f2 = evaluate(self.f2)
        f3 = evaluate(self.f3)

        diff_q_f1 = evaluate(self.diff_q_f1)
        diff_q_f2 = evaluate(self.diff_q_f2)
        diff_q_f3 = evaluate(self.diff_q_f3)

        # Notations: upsilon1 = [Y, 0], upsilon2 = [0, Y]
        #
        L11 = []
        L12 = []
        L21 = []
        L22 = []

        for Tq, diff_Tq in zip(self.chebyshev_t_q, diff_chebyshev_t_q):
           for Tp, diff_Tp in zip(self.chebyshev_t_p, diff_chebyshev_t_p):

                ########################################################################################################
                #
                #   < upsilon1 | Liouvillian | upsilon1 >
                #
                ########################################################################################################

                l11 = evaluate(
                    "-1.j * diff_p_h * diff_Tq * Tp + 1.j * (diff_q_h + diff_q_f3) * Tq * diff_Tp"
                    "-0.5 * (q * diff_q_h + p * diff_p_h - 2. * h + (q * diff_q_f3 - 2. * f3)) * Tq * Tp",
                )
                L11.append(
                    csc_matrix(self._get_coeffs(l11).reshape(-1, 1))
                )

                ########################################################################################################
                #
                #   < upsilon1 | Liouvillian | upsilon2 >
                #
                ########################################################################################################

                l12 = evaluate(
                    "1j * (diff_q_f1 - 1j * diff_q_f2) * Tq * diff_Tp"
                    "-0.5 * (q * diff_q_f1 - 2. * f1 - 1j * (q * diff_q_f2 - 2. * f2)) * Tq * Tp"
                )
                L12.append(
                    csc_matrix(self._get_coeffs(l12).reshape(-1, 1))
                )

                ########################################################################################################
                #
                #   < upsilon2 | Liouvillian | upsilon1 >
                #
                ########################################################################################################

                l21 = evaluate(
                    "1j * (diff_q_f1 + 1j * diff_q_f2) * Tq * diff_Tp"
                    "-0.5 * (q * diff_q_f1 - 2. * f1 + 1j * (q * diff_q_f2 - 2. * f2)) * Tq * Tp"
                )
                L21.append(
                    csc_matrix(self._get_coeffs(l21).reshape(-1, 1))
                )

                ########################################################################################################
                #
                #   < upsilon2 | Liouvillian | upsilon2 >
                #
                ########################################################################################################

                l22 = evaluate(
                    "-1.j * diff_p_h * diff_Tq * Tp + 1.j * (diff_q_h - diff_q_f3) * Tq * diff_Tp"
                    "-0.5 * (q * diff_q_h + p * diff_p_h - 2. * h - (q * diff_q_f3 - 2. * f3)) * Tq * Tp",
                )
                L22.append(
                    csc_matrix(self._get_coeffs(l22).reshape(-1, 1))
                )

        # assemble the matrix of the Liouvillian from the blocks
        # Note that transposition is crucial
        L11 = hstack(L11)
        L12 = hstack(L12)
        L22 = hstack(L22)
        L21 = hstack(L21)

        self.liouvillian = bmat(
            [[L11, L12], [L21, L22]]
        )

    def set_upsilon(self, func_upsilon1:str = "0. * p + 0. * q", func_upsilon2:str = "0. * p + 0. * q") -> object:
        """
        Setting the coefficients of the Chebyshev polynomial expansion to fit the supplied functions.
        :param func_upsilon1: a string to be evaluated by numexpr representing a function of q and p
        :param func_upsilon2: a string to be evaluated by numexpr representing a function of q and p
        :return: self
        """
        p = self.comp_p
        q = self.comp_q

        evaluate(func_upsilon1, out=self.upsilon1_coeff)

        # make sure that that data remains in self.upsilon1_coeff
        # it is important for fast propagation
        self.upsilon1_coeff[:] = self._get_coeffs(self.upsilon1_coeff)

        # using the cosine DFT to get the coefficients
        evaluate(func_upsilon2, out=self.upsilon2_coeff)

        # make sure that that data remains in self.upsilon2_coeff
        # it is important for fast propagation
        self.upsilon2_coeff[:] = self._get_coeffs(self.upsilon2_coeff)

        return self

    def _get_coeffs(self, tmp: np.array) -> np.array:
        """
        using the cosine DFT to get the coefficients in the Chebyshev polynomial expansion
        see, e.g., https://en.wikipedia.org/wiki/Chebyshev_polynomials#Example_1
        :param tmp: array in p and q
        :return: tmp
        """
        tmp = dctn(tmp, overwrite_x=True)

        # fix normalization
        tmp /= self.n_basis_vect ** 2
        tmp[0, :] /= 2.
        tmp[:, 0] /= 2.

        ################################################################################################################
        #
        #   !!!!!!!!!!!!!!!!! careful !!!!!!!!!!!!!!!!!!
        #
        ################################################################################################################
        tmp[np.abs(tmp) < 1e-12] = 0.

        return tmp

    def save_chebyshev(self, q:np.array, p:np.array):
        """
        Pre-calculate the Chebyshev polynomials at specified grid points
        :param p:
        :param q:
        :return: None
        """
        # if p and q were supplied before, do not re-calculate
        if self._p is not p or self._q is not q:

            # consistency checks
            assert len(p.shape) == 2 and (p.shape[0] == 1 or p.shape[1] == 1), "Array p must be flat"
            assert p.min() >= -1. and p.max() <= 1., "Range of p must be [-1, 1]"

            assert len(q.shape) == 2 and (q.shape[0] == 1 or q.shape[1] == 1), "Array q must be flat"
            assert q.min() >= -1. and q.max() <= 1., "Range of q must be [-1, 1]"

            self._p = p
            self._q = q

            self.chebyshev_t_p = get_chebyshev_list(p, np.ones_like(p), p, self.n_basis_vect)
            self.chebyshev_u_p = get_chebyshev_list(p, np.ones_like(p), 2. * p, self.n_basis_vect)

            self.chebyshev_t_q = get_chebyshev_list(q, np.ones_like(q), q, self.n_basis_vect)
            self.chebyshev_u_q = get_chebyshev_list(q, np.ones_like(q), 2. * q, self.n_basis_vect)

            ############################################################################################################
            #
            #   Allocate arrays
            #
            ############################################################################################################

            self.upsilon1 = np.zeros((p.size, q.size), dtype=np.complex)
            self.upsilon2 = self.upsilon1.copy()

            self.diff_p_upsilon1 = self.upsilon1.copy()
            self.diff_q_upsilon1 = self.upsilon1.copy()

            self.diff_p_upsilon2 = self.upsilon2.copy()
            self.diff_q_upsilon2 = self.upsilon2.copy()

            self.D11 = self.upsilon1.copy()
            self.D12 = self.D11.copy()
            self.D22 = self.D11.copy()

            self.classical_rho = np.zeros_like(self.D11, dtype=np.float)

    def get_upsilon(self, q: np.array, p: np.array) -> object:
        """
        Evaluate the Hybrid function at specified phase space points.
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.save_chebyshev(q, p)

        upsilon1 = self.upsilon1
        upsilon1.fill(0.)

        upsilon2 = self.upsilon2
        upsilon2.fill(0.)

        for n, Tq in enumerate(self.chebyshev_t_q):
            for m, Tp in enumerate(self.chebyshev_t_p):

                c = self.upsilon1_coeff[n, m]
                evaluate("upsilon1 + c * Tq * Tp", out=upsilon1)

                c = self.upsilon2_coeff[n, m]
                evaluate("upsilon2 + c * Tq * Tp", out=upsilon2)

        return self

    def get_diff_p_upsilon(self, q: np.array, p: np.array) -> object:
        """
        Evaluate the momentum (p) derivatives of the Hybrid function at specified phase space points.
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.save_chebyshev(q, p)

        diff_p_upsilon1 = self.diff_p_upsilon1
        diff_p_upsilon1.fill(0.)

        diff_p_upsilon2 = self.diff_p_upsilon2
        diff_p_upsilon2.fill(0.)

        for n, Tq in enumerate(self.chebyshev_t_q):
            for m, Up in enumerate(self.chebyshev_u_p[:-1]):

                c = self.upsilon1_coeff[n, m + 1] * (m + 1)
                evaluate("diff_p_upsilon1 + c * Tq * Up", out=diff_p_upsilon1)

                c = self.upsilon2_coeff[n, m + 1] * (m + 1)
                evaluate("diff_p_upsilon2 + c * Tq * Up", out=diff_p_upsilon2)

        return self

    def get_diff_q_upsilon(self, q: np.array, p: np.array) -> object:
        """
        Evaluate the coordinate (q) derivatives of the Hybrid function at specified phase space points.
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.save_chebyshev(q, p)

        diff_q_upsilon1 = self.diff_q_upsilon1
        diff_q_upsilon1.fill(0.)

        diff_q_upsilon2 = self.diff_q_upsilon2
        diff_q_upsilon2.fill(0.)

        for n, Uq in enumerate(self.chebyshev_u_q[:-1]):
            for m, Tp in enumerate(self.chebyshev_t_p):

                c = self.upsilon1_coeff[n + 1, m] * (n + 1)
                evaluate("diff_q_upsilon1 + c * Uq * Tp", out=diff_q_upsilon1)

                c = self.upsilon2_coeff[n + 1, m] * (n + 1)
                evaluate("diff_q_upsilon2 + c * Uq * Tp", out=diff_q_upsilon2)

        return self

    def get_d(self, q: np.array, p: np.array) -> object:
        """
        Calculate the hybrid density matrix
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.get_upsilon(q, p)

        self.get_diff_p_upsilon(q, p)
        self.get_diff_q_upsilon(q, p)

        evaluate(
            "2. * abs(upsilon1) ** 2 + real("
            " upsilon1 * conj(_q * diff_q_upsilon1) + upsilon1 * conj(_p * diff_p_upsilon1) "
            " + 2j * diff_q_upsilon1 * conj(diff_p_upsilon1)"
            ")",
            local_dict=vars(self), out=self.D11
        )

        evaluate(
            "2. * upsilon1 * conj(upsilon2)"
            "+ 1.j * (diff_q_upsilon1 * conj(diff_p_upsilon2) - conj(diff_q_upsilon2) * diff_p_upsilon1)"
            "+ 0.5 * upsilon1 * (conj(_q * diff_q_upsilon2) + conj(_p * diff_p_upsilon2))"
            "+ 0.5 * conj(upsilon2) * (_q * diff_q_upsilon1 + _p * diff_p_upsilon1)",
            local_dict=vars(self), out=self.D12
        )

        evaluate(
            "2. * abs(upsilon2) ** 2 + real("
            " upsilon2 * conj(_q * diff_q_upsilon2) + upsilon2 * conj(_p * diff_p_upsilon2) "
            " + 2j * diff_q_upsilon2 * conj(diff_p_upsilon2)"
            ")",
            local_dict=vars(self), out=self.D22
        )

        return self

    def get_classical_density(self, q: np.array, p: np.array) -> np.array:
        """
        Calculate and return the classical density
        :param q: coordinate grid
        :param p: momentum grid
        :return: classical density (self.classical_rho)
        """
        self.get_d(q, p)

        np.add(self.D11.real, self.D22.real, out=self.classical_rho)

        return self.classical_rho

    def propagate(self) -> object:
        """
        Propagate the hybrid wavefunction by the time dt * nsteps.
        :param nsteps: number of time steps taken
        :return: self
        """
        self._flatten_upsilon += self.dt * self.liouvillian.dot(self._flatten_upsilon)
        return self

    def apply_liouvillian(self, nsteps: int = 1) -> object:
        """
        Apply Liouvillian nstep times
        :param nsteps:
        :return: self
        """
        for _ in range(nsteps):
            self._flatten_upsilon[:] = self.liouvillian.dot(self._flatten_upsilon)

        return self


if __name__ == '__main__':

    ####################################################################################################################
    #
    #   Run some test
    #
    ####################################################################################################################

    # # use only 2 threads
    # from numexpr import set_num_threads
    # set_num_threads(2)

    import matplotlib.pyplot as plt
    from wigner_normalize import WignerSymLogNorm, WignerNormalize

    p = np.linspace(-1., 1, 500)[:, np.newaxis]
    q = np.linspace(-1., 1., 200)[np.newaxis, :]

    # inverse temperature in the initial condition given below
    beta = 300

    # The equilibrium stationary state corresponding to the classical thermal
    # state for a harmonic oscillator with the inverse temperature beta
    Upsilon = "1 / ({r} ** 2) * sqrt(1. - (1. + 0.5 * beta * {r} ** 2) * exp(-0.5 * beta * {r} ** 2) )".format(
        r="sqrt(q ** 2 + p ** 2)",
        theta="arctan2(p, q)"
    )

    # import pickle
    #
    # try:
    #     with open('propagator.pickle', 'rb') as f:
    #         prop = pickle.load(f)
    #     print("The instance of CHybridProp is un-pickled")
    #
        # except IOError:
    #     prop = CHybridProp(
    #         n_basis_vect=200,
    #         h="0.5 * (p ** 2 + q ** 2)",
    #         diff_p_h="p",
    #         diff_q_h="q",
    #     )
    #
    #     with open('propagator.pickle', 'wb') as f:
    #         pickle.dump(prop, f)

    prop = CHybridProp(
        n_basis_vect=200,
        h="0.5 * (p ** 2 + q ** 2)",
        diff_p_h="p",
        diff_q_h="q",
    )

    # prop.set_upsilon(func_upsilon2 = Upsilon)
    prop.set_upsilon(Upsilon)

    img_param = dict(
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin='lower',
        cmap='seismic',
        norm=WignerSymLogNorm(linthresh=1e-10)
    )

    rho = prop.get_classical_density(q, p).copy()

    Y_before = prop._flatten_upsilon.copy()

    plt.subplot(121)
    plt.title("Fitted function")

    plt.imshow(rho, **img_param)
    plt.xlabel("$q$ (a.u.)")
    plt.ylabel("$p$ (a.u.)")
    plt.colorbar()

    plt.subplot(122)
    plt.title("Error")

    # prop.propagate(40)
    # prop.apply_liouvillian(4)

    for _ in range(10):
        prop.propagate()

    rho_ = prop.get_classical_density(q, p).copy()

    print(np.abs(rho_ - rho ).max())

    plt.imshow(rho_, **img_param)
    plt.xlabel("$q$ (a.u.)")
    plt.ylabel("$p$ (a.u.)")
    plt.colorbar()

    Y_after = prop._flatten_upsilon.copy()

    plt.show()