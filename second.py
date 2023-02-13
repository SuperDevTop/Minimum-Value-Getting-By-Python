import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import math

def f(x):
    return -(math.log(2*x[1] - x[0]) + math.log(2*x[0] - x[1])+ math.log(1 - x[0] - x[1]))

x0 = np.array([0.25, 0.25])

constr_func = lambda x: np.array([ 2 * x[1] - x[0],
                                   2 * x[0] - x[1],
                                   1 - x[0] - x[1] ])
nonlin_con = NonlinearConstraint(constr_func, 0, np.inf)

def J(x):
    x1 = -1/(2 * x[1] - x[0]) + 2/(2 * x[0] - x[1]) - 1/(1 - x[0] - x[1])
    x2 = 2/(2 * x[1] - x[0]) - 1/(2 * x[0] - x[1]) - 1/(1 - x[0] - x[1])
    return np.array([x1, x2])

res = minimize(f, x0, method = 'trust-constr', jac = J , tol = 1e-10,  constraints = nonlin_con)

print(res)