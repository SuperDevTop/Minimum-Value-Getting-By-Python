
import numpy as np
from scipy.optimize import minimize
import math

def f(x):
    # return 3* x [0]**2 + x [1]**4
    return (x[0] ** 2 - (3 * (x[1] ** 2))) ** 2 + (math.sin(x[0] ** 2 + x[1] ** 2)) ** 2
# specify initial point x0 as a 1-D ndarray
x0 = np.array([1 , 0])

def J(x):
    # return np.array ([6*x[0] , 4* x [1]**3])
    return np.array([4 * x[0]* (x[0] ** 2 - 3 * x[1] ** 2) + 2 * math.sin( x[0] ** 2 + x[1] ** 2) \
            * math.cos(x[0] ** 2 + x[1] ** 2) * 2 * x[0], -6 * x[1] * (x[0] ** 2 - 3 * x[1] ** 2) + 2 * math.sin( x[0] ** 2 + x[1] ** 2) \
            * math.cos(x[0] ** 2 + x[1] ** 2) * 2 * x[1]])

# specify the objective fn as a callable fn , initial point ,
# jacobian fn as a callable fn and the tolerance for termination
res = minimize(f, x0, method = 'BFGS', jac = J , tol = 1e-10)

print(res)