import numpy as np
import scipy as scp


def solve_least_squares(A, b, method):
    if method == "numpy":
        return solve_least_squares_numpy(A, b)
    elif method == "cholesky":
        return solve_least_squares_cholesky(A, b)
    raise RuntimeError("Unknown method")


def solve_least_squares_numpy(A, b):
    return np.linalg.lstsq(A, b, rcond=-1)[0]


def solve_least_squares_cholesky(A, b):
    R = scp.linalg.cholesky(A.T @ A, lower=False)
    y = solve_equation_lower_back_substitution(R.T, A.T @ b)
    return solve_equation_lower_back_substitution(R, y)


def solve_equation_lower_back_substitution(A, b):
    y = np.zeros(A.shape[1])
    y[0] = b[0] / A[0, 0]
    for i in range(1, A.shape[1]):
        y[i] = (b[i] - np.sum([A[i, j] * y[j] for j in range(0, i)])) / A[i, i]
    return y
