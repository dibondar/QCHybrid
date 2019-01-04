import numpy as np
from numexpr import evaluate
from scipy.fftpack import dctn


def get_chebyshev_list(x:np.array, chebyshev0:np.array, chebyshev1:np.array, n_basis_vect:int) -> list:
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


def enumerate_previous_and_next(x: iter):
    """
    Generator to yield (0, 0., x[1]) at the first call, (1, x[0], x[2]) at the second etc.
    :param x: iterable
    :return: three elements
    """
    x = iter(x)
    x_n_minus_1 = next(x)
    x_n = next(x)

    yield 0, 0., x_n

    n = 1

    for x_n_plus_1 in x:

        yield n, x_n_minus_1, x_n_plus_1

        x_n_minus_1, x_n = x_n, x_n_plus_1

        n += 1


class CHybridProp(object):
    """
    The numerical propagator for the hybrid equation of motion using the Chebyshev polynomials as basis.

        upsilon1 = \sum_{n,m=1}^{n_basis_vect - 1} upsilon1_coeff[n, m] T[n](q)T[m](p)

    """
    def __init__(self, *, n_basis_vect:int = 20):
        """
        :param n_basis_vect: the number of the Chebyshev polynomials to use
        :param p: the momentum
        """

        # Saving parameters
        self.n_basis_vect = n_basis_vect

        # allocate the array of coefficients for hybrid wavefunctions
        self.upsilon1_coeff = np.zeros((self.n_basis_vect, self.n_basis_vect), dtype=np.complex)
        self.upsilon2_coeff = self.upsilon1_coeff.copy()

        # variables for pre-calculating Chebyshev polynomials
        self._p = None
        self._q = None

    def set_upsilon(self, func_upsilon1:str = "0. * p + 0. * q", func_upsilon2:str = "0. * p + 0. * q") -> object:
        """
        Setting the coefficients of the Chebyshev polynomial expansion to fit the supplied functions.
        :param func_upsilon1: a string to be evaluated by numexpr representing a function of q and p
        :param func_upsilon2: a string to be evaluated by numexpr representing a function of q and p
        :return: self
        """
        # generate grid points
        chebyshev_zeros = np.cos(np.pi * (np.arange(self.n_basis_vect) + 0.5) / self.n_basis_vect)
        p = chebyshev_zeros[np.newaxis, :]
        q = chebyshev_zeros[:, np.newaxis]

        # using the cosine DFT to get the coefficients
        # see, e.g., https://en.wikipedia.org/wiki/Chebyshev_polynomials#Example_1
        evaluate(func_upsilon1, out=self.upsilon1_coeff)
        self.upsilon1_coeff = dctn(self.upsilon1_coeff, overwrite_x=True)

        # fix normalization
        self.upsilon1_coeff *= (1 / self.n_basis_vect) ** 2
        self.upsilon1_coeff[0, :] /= 2.
        self.upsilon1_coeff[:, 0] /= 2.

        # using the cosine DFT to get the coefficients
        evaluate(func_upsilon2, out=self.upsilon2_coeff)
        self.upsilon2_coeff = dctn(self.upsilon2_coeff, overwrite_x=True)

        # fix normalization
        self.upsilon2_coeff *= (1 / self.n_basis_vect) ** 2
        self.upsilon2_coeff[0, :] /= 2.
        self.upsilon2_coeff[:, 0] /= 2.

        return self

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
            #   Allocate the arrays
            #
            ############################################################################################################

            self.upsilon1 = np.zeros((p.size, q.size), dtype=np.complex)
            self.upsilon2 = self.upsilon1.copy()

            self.diff_p_upsilon1 = self.upsilon1.copy()
            self.diff_q_upsilon1 = self.upsilon1.copy()
            self.q_diff_q_upsilon1 = self.upsilon1.copy()
            self.p_diff_p_upsilon1 = self.upsilon1.copy()

            self.diff_p_upsilon2 = self.upsilon2.copy()
            self.diff_q_upsilon2 = self.upsilon2.copy()
            self.q_diff_q_upsilon2 = self.upsilon2.copy()
            self.p_diff_p_upsilon2 = self.upsilon2.copy()

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

    def get_q_diff_q_upsilon(self, q: np.array, p: np.array) -> object:
        """
        Evaluate the coordinate (q) times derivatives w.r.t. q of the Hybrid function at specified phase space points.
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.save_chebyshev(q, p)

        q_diff_q_upsilon1 = self.q_diff_q_upsilon1
        q_diff_q_upsilon1.fill(0.)

        q_diff_q_upsilon2 = self.q_diff_q_upsilon2
        q_diff_q_upsilon2.fill(0.)

        for n, Uq_n_minus_1, Uq_n_plus_1 in enumerate_previous_and_next(self.chebyshev_u_q[:-1]):
            for m, Tp in enumerate(self.chebyshev_t_p):

                c = self.upsilon1_coeff[n + 1, m] * (n + 1) / 2.
                evaluate("q_diff_q_upsilon1 + c * (Uq_n_plus_1 + Uq_n_minus_1) * Tp", out=q_diff_q_upsilon1)

                c = self.upsilon2_coeff[n + 1, m] * (n + 1) / 2.
                evaluate("q_diff_q_upsilon2 + c * (Uq_n_plus_1 + Uq_n_minus_1) * Tp", out=q_diff_q_upsilon2)

        return self

    def get_p_diff_p_upsilon(self, q: np.array, p: np.array) -> object:
        """
        Evaluate the momentum (p) times derivatives w.r.t. p of the Hybrid function at specified phase space points.
        :param q: coordinate grid
        :param p: momentum grid
        :return: self
        """
        self.save_chebyshev(q, p)

        p_diff_p_upsilon1 = self.p_diff_p_upsilon1
        p_diff_p_upsilon1.fill(0.)

        p_diff_p_upsilon2 = self.p_diff_p_upsilon2
        p_diff_p_upsilon2.fill(0.)

        for n, Tq in enumerate(self.chebyshev_t_q):
            for m, Up_m_minus_1, Up_m_plus_1 in enumerate_previous_and_next(self.chebyshev_u_p[:-1]):

                c = self.upsilon1_coeff[n, m + 1] * (m + 1) / 2.
                evaluate("p_diff_p_upsilon1 + c * (Up_m_plus_1 + Up_m_minus_1) * Tq", out=p_diff_p_upsilon1)

                c = self.upsilon2_coeff[n, m + 1] * (m + 1) / 2.
                evaluate("p_diff_p_upsilon2 + c * (Up_m_plus_1 + Up_m_minus_1) * Tq", out=p_diff_p_upsilon2)

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

        self.get_q_diff_q_upsilon(q, p)
        self.get_p_diff_p_upsilon(q, p)

        evaluate(
            "2. * abs(upsilon1) ** 2 + real("
            " upsilon1 * conj(q_diff_q_upsilon1) + upsilon1 * conj(p_diff_p_upsilon1) "
            " + 2j * diff_q_upsilon1 * conj(diff_p_upsilon1)"
            ")",
            local_dict=vars(self), out=self.D11
        )

        evaluate(
            "2. * upsilon1 * conj(upsilon2)"
            "+ 1.j * (diff_q_upsilon1 * conj(diff_p_upsilon2) - conj(diff_q_upsilon2) * diff_p_upsilon1)"
            "+ 0.5 * upsilon1 * (conj(q_diff_q_upsilon2) + conj(p_diff_p_upsilon2))"
            "+ 0.5 * conj(upsilon2) * (q_diff_q_upsilon1 + p_diff_p_upsilon1)",
            local_dict=vars(self), out=self.D12
        )

        evaluate(
            "2. * abs(upsilon2) ** 2 + real("
            " upsilon2 * conj(q_diff_q_upsilon2) + upsilon2 * conj(p_diff_p_upsilon2) "
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
    beta = 100

    # The equilibrium stationary state corresponding to the classical thermal
    # state for a harmonic oscillator with the inverse temperature beta
    Upsilon = "1 / ({r} ** 2) * sqrt(1. - (1. + 0.5 * beta * {r} ** 2) * exp(-0.5 * beta * {r} ** 2) )".format(
        r="sqrt(q ** 2 + p ** 2)",
        theta="arctan2(p, q)"
    )

    exact_rho = evaluate("exp(-0.5 * beta * (p ** 2 + q ** 2))")

    prop = CHybridProp(n_basis_vect=150).set_upsilon(func_upsilon2 = Upsilon)
    # prop = CHybridProp(n_basis_vect=150).set_upsilon(Upsilon)

    img_param = dict(
        extent=[q.min(), q.max(), p.min(), p.max()],
        origin='lower',
        cmap='seismic',
        norm=WignerSymLogNorm(linthresh=1e-10)
    )

    rho = prop.get_classical_density(q, p)

    plt.subplot(121)
    plt.title("Fitted function")

    plt.imshow(rho, **img_param)
    plt.xlabel("$q$ (a.u.)")
    plt.ylabel("$p$ (a.u.)")
    plt.colorbar()

    print(np.sum(rho < 0))

    plt.subplot(122)
    plt.title("Error")

    plt.imshow(np.abs(exact_rho / exact_rho.max() - rho / rho.max()), **img_param)
    plt.xlabel("$q$ (a.u.)")
    plt.ylabel("$p$ (a.u.)")
    plt.colorbar()

    plt.show()
