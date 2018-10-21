import numpy as np
from QuantumClassicalDynamics.split_op_pauli_like1D import SplitOpPauliLike1D

class CAnalyticQCHybrid(object):
    """
    Equations for plotting the found analytical solution

    hbar = 1, m = 1, omega = 1, beta = 0.95,
    """
    def __init__(self, p, q, kT):
        """
        Constructor
        :param p:
        :param q:
        :param kT: temperature
        :return: None
        """
        self.p = p
        self.q = q

        # extract the volume element in the phase space
        p = p.reshape(-1)
        q = q.reshape(-1)

        self.dq = q[1] - q[0]
        self.dp = p[1] - p[0]

        self.dqdp = self.dq * self.dp

        # save the temperature
        self.kT = kT

    def D11(self, t):
        """
        The 11 element of the density matrix
        :param t: time
        :return: numpy.array
        """
        p = self.p
        q = self.q
        kT = self.kT

        #############################################################################################
        #
        # Code generated by Maple (exact_solution.mw) begins
        #
        #############################################################################################

        t3 = np.sqrt(0.1e1 / np.pi * kT)
        t4 = q ** 2
        t6 = p ** 2
        t7 = t4 / 20 + t6
        t8 = np.sqrt(20)
        t11 = np.arctan2(q * t8 / 20, p)
        t14 = -t11 + t8 * t / 20
        t15 = np.cos(t14)
        t16 = t15 ** 2
        t17 = t16 * t7
        t19 = np.sin(t14)
        t20 = t19 ** 2
        t21 = t20 * t7
        t24 = 0.1e1 / kT
        t25 = t24 * (t17 / 2 + 10 * t21)
        t26 = 1 + t25
        t27 = np.exp(-t25)
        t30 = np.sqrt(-t27 * t26 + 1)
        t33 = t17 + 20 * t21
        t34 = 0.1e1 / t33
        t37 = 0.39e2 / 0.20e2 * t4 + t6
        t38 = np.sqrt(39)
        t39 = t8 * t38
        t42 = np.arctan2(q * t39 / 20, p)
        t46 = -t42 + t8 * t38 * t / 20
        t47 = np.cos(t46)
        t48 = t47 ** 2
        t49 = t48 * t37
        t51 = np.sin(t46)
        t52 = t51 ** 2
        t53 = t52 * t37
        t56 = t24 * (t49 / 2 + 0.10e2 / 0.39e2 * t53)
        t57 = 1 + t56
        t58 = np.exp(-t56)
        t61 = np.sqrt(-t58 * t57 + 1)
        t64 = t49 + 0.20e2 / 0.39e2 * t53
        t65 = 0.1e1 / t64
        t68 = abs(t34 * t30 * t3 + t65 * t61 * t3)
        t69 = t68 ** 2
        t70 = np.sqrt(2)
        t71 = t3 * t70
        t77 = t34 * t30 * t71 / 2 + t65 * t61 * t71 / 2
        t79 = t33 ** 2
        t80 = 0.1e1 / t79
        t81 = (t30).conjugate()
        t82 = t81 * t80
        t83 = t16 * q
        t86 = t8 * t15 * t7
        t87 = 0.1e1 / p
        t88 = 0.1e1 / t6
        t89 = t88 * t4
        t92 = 0.1e1 / (1 + t89 / 20)
        t95 = t19 * t92 * t87 * t86
        t97 = t20 * q
        t99 = t83 / 10 - 0.19e2 / 0.10e2 * t95 + 2 * t97
        t104 = 0.1e1 / t30 * t34
        t107 = t83 / 20 - 0.19e2 / 0.20e2 * t95 + t97
        t111 = t27 * t24
        t116 = (t111 * t107 * t26 - t27 * t24 * t107) * t104 * t71 / 4
        t117 = t64 ** 2
        t118 = 0.1e1 / t117
        t119 = (t61).conjugate()
        t120 = t119 * t118
        t121 = t48 * q
        t123 = t47 * t37
        t129 = t51 / (1 + 0.39e2 / 0.20e2 * t89)
        t131 = t129 * t87 * t8 * t38 * t123
        t133 = t52 * q
        t135 = 0.39e2 / 0.10e2 * t121 + 0.19e2 / 0.390e3 * t131 + 2 * t133
        t140 = 0.1e1 / t61 * t65
        t143 = 0.39e2 / 0.20e2 * t121 + 0.19e2 / 0.780e3 * t131 + t133
        t147 = t58 * t24
        t152 = (t147 * t143 * t57 - t58 * t24 * t143) * t140 * t71 / 4
        t156 = t16 * p
        t158 = t88 * q
        t161 = t19 * t92 * t158 * t86
        t163 = t20 * p
        t171 = t156 + 0.19e2 / 0.20e2 * t161 + 20 * t163
        t180 = t48 * p
        t184 = t129 * t158 * t39 * t123
        t186 = t52 * p
        t194 = t180 - 0.19e2 / 0.780e3 * t184 + 0.20e2 / 0.39e2 * t186
        t203 = -(2 * t156 + 0.19e2 / 0.10e2 * t161 + 40 * t163) * t82 * t71 / 2 + (
                    t111 * t171 * t26 - t27 * t24 * t171) * t104 * t71 / 4 - (
                           2 * t180 - 0.19e2 / 0.390e3 * t184 + 0.40e2 / 0.39e2 * t186) * t120 * t71 / 2 + (
                           t147 * t194 * t57 - t58 * t24 * t194) * t140 * t71 / 4
        t217 = np.real((-t99 * t82 * t71 / 2 + t116 - t135 * t120 * t71 / 2 + t152) * q * t77 + t203 * p * t77 + complex(0,
                                                                                                                    2) * t203 * (
                              t116 - t99 * t80 * t30 * t71 / 2 + t152 - t135 * t118 * t61 * t71 / 2))
        self._D11 = t69 + t217

        #############################################################################################
        #
        # Code generated by Maple ends
        #
        #############################################################################################

        return self._D11

    def D12(self, t):
        """
        The 12 element of the hybrid density matrix
        :param t: time
        :return: numpy.array
        """
        p = self.p
        q = self.q
        kT = self.kT

        #############################################################################################
        #
        # Code generated by Maple (exact_solution.mw) begins
        #
        #############################################################################################

        t1 = np.sqrt(2)
        t4 = np.sqrt(0.1e1 / np.pi * kT)
        t5 = t4 * t1
        t6 = q ** 2
        t8 = p ** 2
        t9 = t6 / 20 + t8
        t10 = np.sqrt(20)
        t13 = np.arctan2(q * t10 / 20, p)
        t16 = -t13 + t10 * t / 20
        t17 = np.cos(t16)
        t18 = t17 ** 2
        t19 = t18 * t9
        t21 = np.sin(t16)
        t22 = t21 ** 2
        t23 = t22 * t9
        t26 = 0.1e1 / kT
        t27 = t26 * (t19 / 2 + 10 * t23)
        t28 = 1 + t27
        t29 = np.exp(-t27)
        t32 = np.sqrt(-t29 * t28 + 1)
        t34 = t19 + 20 * t23
        t35 = 0.1e1 / t34
        t37 = t35 * t32 * t5
        t39 = 0.39e2 / 0.20e2 * t6 + t8
        t40 = np.sqrt(39)
        t41 = t10 * t40
        t44 = np.arctan2(q * t41 / 20, p)
        t48 = -t44 + t10 * t40 * t / 20
        t49 = np.cos(t48)
        t50 = t49 ** 2
        t51 = t50 * t39
        t53 = np.sin(t48)
        t54 = t53 ** 2
        t55 = t54 * t39
        t58 = t26 * (t51 / 2 + 0.10e2 / 0.39e2 * t55)
        t59 = 1 + t58
        t60 = np.exp(-t58)
        t63 = np.sqrt(-t60 * t59 + 1)
        t65 = t51 + 0.20e2 / 0.39e2 * t55
        t66 = 0.1e1 / t65
        t68 = t66 * t63 * t5
        t70 = t37 / 2 + t68 / 2
        t73 = (t68 / 2 - t37 / 2).conjugate()
        t77 = 0.1e1 / t32 * t35
        t78 = t18 * q
        t81 = t10 * t17 * t9
        t82 = 0.1e1 / p
        t83 = 0.1e1 / t8
        t84 = t83 * t6
        t87 = 0.1e1 / (1 + t84 / 20)
        t90 = t21 * t87 * t82 * t81
        t92 = t22 * q
        t93 = t78 / 20 - 0.19e2 / 0.20e2 * t90 + t92
        t97 = t29 * t26
        t102 = (-t29 * t26 * t93 + t97 * t93 * t28) * t77 * t5 / 4
        t103 = t34 ** 2
        t104 = 0.1e1 / t103
        t105 = t104 * t32
        t109 = t78 / 10 - 0.19e2 / 0.10e2 * t90 + 2 * t92
        t114 = 0.1e1 / t63 * t66
        t115 = t50 * q
        t117 = t49 * t39
        t123 = t53 / (1 + 0.39e2 / 0.20e2 * t84)
        t125 = t123 * t82 * t10 * t40 * t117
        t127 = t54 * q
        t128 = 0.39e2 / 0.20e2 * t115 + 0.19e2 / 0.780e3 * t125 + t127
        t132 = t60 * t26
        t137 = (t132 * t128 * t59 - t60 * t26 * t128) * t114 * t5 / 4
        t138 = t65 ** 2
        t139 = 0.1e1 / t138
        t140 = t139 * t63
        t144 = 0.39e2 / 0.10e2 * t115 + 0.19e2 / 0.390e3 * t125 + 2 * t127
        t148 = t102 - t109 * t105 * t5 / 2 + t137 - t144 * t140 * t5 / 2
        t149 = (t63).conjugate()
        t150 = t149 * t139
        t151 = t50 * p
        t154 = t83 * q
        t156 = t123 * t154 * t41 * t117
        t158 = t54 * p
        t160 = 2 * t151 - 0.19e2 / 0.390e3 * t156 + 0.40e2 / 0.39e2 * t158
        t166 = t151 - 0.19e2 / 0.780e3 * t156 + 0.20e2 / 0.39e2 * t158
        t174 = (t132 * t166 * t59 - t60 * t26 * t166) * t114 * t5 / 4
        t175 = (t32).conjugate()
        t176 = t175 * t104
        t177 = t18 * p
        t181 = t21 * t87 * t154 * t81
        t183 = t22 * p
        t185 = 2 * t177 + 0.19e2 / 0.10e2 * t181 + 40 * t183
        t191 = t177 + 0.19e2 / 0.20e2 * t181 + 20 * t183
        t199 = (-t29 * t26 * t191 + t97 * t191 * t28) * t77 * t5 / 4
        t200 = -t160 * t150 * t5 / 2 + t174 + t185 * t176 * t5 / 2 - t199
        t208 = -t144 * t150 * t5 / 2 + t137 + t109 * t176 * t5 / 2 - t102
        t215 = t199 - t185 * t105 * t5 / 2 + t174 - t160 * t140 * t5 / 2
        self._D12 = 2 * t73 * t70 + complex(0, 1) * (t200 * t148 - t215 * t208) + (t200 * p + t208 * q) * t70 / 2 + (
                    t215 * p + t148 * q) * t73 / 2

        #############################################################################################
        #
        # Code generated by Maple ends
        #
        #############################################################################################

        return self._D12

    def D22(self, t):
        """
        The 22 element of the hybrid density matrix
        :param t: time
        :return: numpy.array
        """
        p = self.p
        q = self.q
        kT = self.kT

        #############################################################################################
        #
        # Code generated by Maple (exact_solution.mw) begins
        #
        #############################################################################################

        t3 = np.sqrt(0.1e1 / np.pi * kT)
        t4 = q ** 2
        t6 = p ** 2
        t7 = 0.39e2 / 0.20e2 * t4 + t6
        t8 = np.sqrt(39)
        t9 = np.sqrt(20)
        t10 = t9 * t8
        t13 = np.arctan2(q * t10 / 20, p)
        t17 = -t13 + t9 * t8 * t / 20
        t18 = np.cos(t17)
        t19 = t18 ** 2
        t20 = t19 * t7
        t22 = np.sin(t17)
        t23 = t22 ** 2
        t24 = t23 * t7
        t27 = 0.1e1 / kT
        t28 = t27 * (t20 / 2 + 0.10e2 / 0.39e2 * t24)
        t29 = 1 + t28
        t30 = np.exp(-t28)
        t33 = np.sqrt(-t30 * t29 + 1)
        t36 = t20 + 0.20e2 / 0.39e2 * t24
        t37 = 0.1e1 / t36
        t40 = t4 / 20 + t6
        t43 = np.arctan2(q * t9 / 20, p)
        t46 = -t43 + t9 * t / 20
        t47 = np.cos(t46)
        t48 = t47 ** 2
        t49 = t48 * t40
        t51 = np.sin(t46)
        t52 = t51 ** 2
        t53 = t52 * t40
        t56 = t27 * (t49 / 2 + 10 * t53)
        t57 = 1 + t56
        t58 = np.exp(-t56)
        t61 = np.sqrt(-t58 * t57 + 1)
        t64 = t49 + 20 * t53
        t65 = 0.1e1 / t64
        t68 = abs(t37 * t33 * t3 - t65 * t61 * t3)
        t69 = t68 ** 2
        t70 = np.sqrt(2)
        t71 = t3 * t70
        t77 = t37 * t33 * t71 / 2 - t65 * t61 * t71 / 2
        t79 = t36 ** 2
        t80 = 0.1e1 / t79
        t81 = (t33).conjugate()
        t82 = t81 * t80
        t83 = t19 * q
        t85 = t18 * t7
        t87 = 0.1e1 / p
        t89 = 0.1e1 / t6
        t90 = t89 * t4
        t94 = t22 / (1 + 0.39e2 / 0.20e2 * t90)
        t96 = t94 * t87 * t9 * t8 * t85
        t98 = t23 * q
        t100 = 0.39e2 / 0.10e2 * t83 + 0.19e2 / 0.390e3 * t96 + 2 * t98
        t105 = 0.1e1 / t33 * t37
        t108 = 0.39e2 / 0.20e2 * t83 + 0.19e2 / 0.780e3 * t96 + t98
        t112 = t30 * t27
        t117 = (t112 * t108 * t29 - t30 * t27 * t108) * t105 * t71 / 4
        t118 = t64 ** 2
        t119 = 0.1e1 / t118
        t120 = (t61).conjugate()
        t121 = t120 * t119
        t122 = t48 * q
        t125 = t9 * t47 * t40
        t128 = 0.1e1 / (1 + t90 / 20)
        t131 = t51 * t128 * t87 * t125
        t133 = t52 * q
        t135 = t122 / 10 - 0.19e2 / 0.10e2 * t131 + 2 * t133
        t140 = 0.1e1 / t61 * t65
        t143 = t122 / 20 - 0.19e2 / 0.20e2 * t131 + t133
        t147 = t58 * t27
        t152 = (t147 * t143 * t57 - t58 * t27 * t143) * t140 * t71 / 4
        t156 = t19 * p
        t159 = t89 * q
        t161 = t94 * t159 * t10 * t85
        t163 = t23 * p
        t171 = t156 - 0.19e2 / 0.780e3 * t161 + 0.20e2 / 0.39e2 * t163
        t180 = t48 * p
        t184 = t51 * t128 * t159 * t125
        t186 = t52 * p
        t194 = t180 + 0.19e2 / 0.20e2 * t184 + 20 * t186
        t203 = -(2 * t156 - 0.19e2 / 0.390e3 * t161 + 0.40e2 / 0.39e2 * t163) * t82 * t71 / 2 + (
                    t112 * t171 * t29 - t30 * t27 * t171) * t105 * t71 / 4 + (
                           2 * t180 + 0.19e2 / 0.10e2 * t184 + 40 * t186) * t121 * t71 / 2 - (
                           t147 * t194 * t57 - t58 * t27 * t194) * t140 * t71 / 4
        t217 = np.real((-t100 * t82 * t71 / 2 + t117 + t135 * t121 * t71 / 2 - t152) * q * t77 + t203 * p * t77 + complex(0,
                                                                                                                     2) * t203 * (
                              t117 - t100 * t80 * t33 * t71 / 2 - t152 + t135 * t119 * t61 * t71 / 2))
        self._D22 = t69 + t217

        #############################################################################################
        #
        # Code generated by Maple ends
        #
        #############################################################################################

        return self._D22

    def calculate_D(self, t):
        """
        Save the components of the hybrid density matrix.
        The method must be called before all the other methods are called.
        :param t: time
        :return: self
        """
        self.D11(t)
        self.D12(t)
        self.D22(t)
        return self

    def classical_density(self):
        """
        Calculate the classical. This method must be called after self.calculate_D().
        :return: numpy.array
        """
        rho = self._D11 + self._D22

        # Check for the normalization condition
        assert np.allclose(rho.sum() * self.dqdp, 1), "Classical density must be normalized."

        # Check for the positivity of the density
        assert np.allclose(rho[rho < 0], 0), "Classical density mus be nonnegative"

        assert not np.isnan(rho).any(), "Classical density contains NaNs."

        # Make sure all values are real
        assert isinstance(rho[0, 0],  np.float), "Classical density must be real"

        return rho

    def quantum_density(self):
        """
        Get the quantum density matrix. This method must be called after self.calculate_D().
        :return: 2x2 numpy.array
        """
        d12 = self._D12.sum()

        rho = np.array(
            [[self._D11.sum(), d12], [d12.conjugate(), self._D22.sum()]]
        )

        rho *= self.dqdp

        # For the following consistency checks, get eigenvalues
        p = np.linalg.eigvalsh(rho)

        assert np.allclose(p[p < 0], 0), "Quantum density matrix must be non-negative"

        assert np.allclose(p.sum(), 1), "Trace of quantum density matrix must be one"

        return rho

    def quantum_purity(self):
        """
        Get the purity of the quantum density matrix. This method must be called after self.calculate_D().
        :return: float
        """
        rho = self.quantum_density()

        return rho.dot(rho).trace().real

########################################################################################################
#
#  Plots
#
########################################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from wigner_normalize import WignerNormalize, WignerSymLogNorm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    # Bloch sphere drawing tool
    from qutip import Bloch, Qobj

    ########################################################################################################
    #
    # Make animation
    #
    ########################################################################################################

    class CVisualizeAnim(object):
        """
        Class for drawing animation
        """
        def __init__(self, fig):

            #################################################################
            #
            # Initialize plotting exact plots tools
            #
            #################################################################

            # p = np.linspace(-20., 20., 1000)[:, np.newaxis]
            # q = np.linspace(-20., 20., 1000)[np.newaxis, :]
            #
            # self.hybrid = CAnalyticQCHybrid(p, q, kT=0.5)
            #
            # img_params = dict(
            #     extent=[q.min(), q.max(), p.min(), p.max()],
            #     origin='lower',
            #     cmap='seismic',
            #     # norm=WignerNormalize(vmin=-0.1, vmax=0.1)
            #     norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.01, vmax=0.1)
            # )

            p = np.linspace(-0.1, 0.1, 500)[:, np.newaxis]
            q = np.linspace(-0.1, 0.1, 500)[np.newaxis, :]

            self.hybrid = CAnalyticQCHybrid(p, q, kT=1e-5)

            img_params = dict(
                extent=[q.min(), q.max(), p.min(), p.max()],
                origin='lower',
                cmap='seismic',
                # norm=WignerNormalize(vmin=-0.1, vmax=0.1)
                norm=WignerSymLogNorm(linthresh=1e-10, vmin=-0.01, vmax=0.1)
            )

            #################################################################
            #
            # Initialize Pauli propagator
            #
            #################################################################

            self.pauli = SplitOpPauliLike1D(
                X_amplitude=2 * q.max(),
                X_gridDIM=2 * 1024,
                dt=0.0005,
                K0="0.5 * P ** 2",
                V0="0.5 * X ** 2",
                V1="0.5 * 0.95 * X ** 2",

                # kT=self.hybrid.kT, # parameters controlling the width of the initial wavepacket
            ).set_wavefunction("exp(-0.5 * X ** 2)")

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            self.fig.suptitle(
                "Quantum-classical hybrid $m=1$, $\omega=1$, $\\beta=0.95$, $kT={:.1e}$ (a.u.) (a.u.)".format(self.hybrid.kT)
            )

            self.ax = fig.add_subplot(221)
            self.ax.set_title('Classical density')

            # generate empty plots
            self.img_classical_density = self.ax.imshow([[0]], **img_params)

            self.ax.set_xlabel('$q$ (a.u.)')
            self.ax.set_ylabel('$p$ (a.u.)')

            ax = fig.add_subplot(222)

            ax.set_title('Quantum purity')
            self.quantum_purity_plot, = ax.plot([0., 40], [1, 0.5])
            ax.set_xlabel('time (a.u.)')
            ax.set_ylabel("quantum purity")

            self.time = []
            self.qpurity = []

            ax = fig.add_subplot(223)

            ax.set_title("Coordinate distribution")
            self.c_coordinate_distribution, = ax.semilogy(
                [self.hybrid.q.min(), self.hybrid.q.max()], [1e-11, 1e0], label="hybrid"
            )
            self.pauli_coordinate_distribution, = ax.semilogy(
                [self.hybrid.q.min(), self.hybrid.q.max()], [1e-11, 1e0], label="Pauli"
            )

            ax.legend()

            ax.set_xlabel('$q$ (a.u.)')
            ax.set_ylabel('Probability density')

            ax = fig.add_subplot(224, projection='3d', azim=0,elev=0)

            self.bloch = Bloch(axes=ax)
            self.bloch.make_sphere()

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # convert the frame number to time
            t = 0.05 * frame_num
            # t= self.pauli.t

            # calculate the hybrid density matrix
            self.hybrid.calculate_D(t)

            # plot the classical density
            c_rho = self.hybrid.classical_density()

            self.img_classical_density.set_array(c_rho)
            self.ax.set_title('Classical density \n $t = {:.1f}$ (a.u.)'.format(t))

            # plot the coordinate distribution for the classical density
            coordinate_marginal = c_rho.sum(axis=0)
            # coordinate_marginal *= self.hybrid.dp
            coordinate_marginal /= coordinate_marginal.max()

            self.c_coordinate_distribution.set_data(self.hybrid.q.reshape(-1), coordinate_marginal)
            coordinate_density = self.pauli.coordinate_density
            coordinate_density /= coordinate_density.max()
            self.pauli_coordinate_distribution.set_data(self.pauli.X, coordinate_density)

            # plot quantum purity
            self.time.append(t)
            self.qpurity.append(self.hybrid.quantum_purity())

            self.quantum_purity_plot.set_data(self.time, self.qpurity)

            # plot Bloch vector
            self.bloch.clear()
            self.bloch.add_states(
                Qobj(self.hybrid.quantum_density())
            )
            self.bloch.make_sphere()

            #
            # self.pauli.propagate(100)

            return self.img_classical_density, self.quantum_purity_plot, self.bloch, self.pauli_coordinate_distribution


    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    visualizer = CVisualizeAnim(fig)
    animation = FuncAnimation(
        fig, visualizer, frames=np.arange(400), repeat=True, # blit=True
    )
    plt.show()

    # If you want to make a movie, comment "plt.show()" out and uncomment the lines bellow
    # Save animation into the file
    # animation.save('hybrid_animation.mp4')

    # check weather the energy is preserved during the
    energy = np.array(visualizer.pauli.hamiltonian_average)
    print("Hamiltonian preserved: {:.1e} %".format(np.real(1 - energy.min() / energy.max()) * 100.))

    #######################################################################################################

    #plt.imshow(hybrid.classical_density(0.5), **img_params)
    #plt.ylabel('$p$ (a.u.)')
    #plt.xlabel('$q$ (a.u.)')
    #plt.show()