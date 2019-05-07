import numpy as np
from sympy import Integer, RealNumber, lambdify, simplify


def symbolic_exp(init_state, operator, t, var_vals, rel_epsilon=1e-5):
    """
    exp(-1j * t * operator) init_state =
        sum( (-1j * dt)^k / k! * operator^k init_state, k=0..n)

    :param init_state:
    :param operator:
    :param n:
    :return:
    """

    minus_jt = -1j * RealNumber(t)

    current_term = init_state

    result = lambdify(init_state.free_symbols, init_state, "numpy")(
        **{str(key): var_vals[str(key)] for key in init_state.free_symbols}
    )

    # make sure the result is a numpy array
    result = np.array(result, copy=False)

    delta = np.array([np.inf])

    # counter
    k = Integer(0)

    # loop till convergence
    while np.linalg.norm(delta.reshape(-1), np.inf) / np.linalg.norm(result.reshape(-1), np.inf) > rel_epsilon:

        current_term = operator(current_term)
        k += 1
        current_term *= minus_jt / k
        current_term = simplify(current_term)

        delta = lambdify(current_term.free_symbols, current_term, "numpy")(
            **{str(key): var_vals[str(key)] for key in current_term.free_symbols}
        )

        if k == 1:
            result = result + delta
        else:
            result += delta

        print(k)
        print(np.linalg.norm(delta.reshape(-1), np.inf))

    return result