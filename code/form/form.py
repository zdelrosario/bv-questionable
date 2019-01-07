"""
Subroutines for FORM.

Rick Fenrich 10/22/18
MIB addition: ZDR 10/29/18
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize as minimize

def pma(
        func,
        gradFunc,
        u0,
        pf,
        tfmJac,
        That,
        C = 0.95,
        n_restarts = 1,
        tol = 1e-6,
        niter = 100
):
    """
    Do inverse reliability analysis with FORM.

    Arguments:
    func: function which takes a 1-D array of standard normal variables u and
        returns the limit state value g(u)
    gradFunc: function which takes a 1-D array of standard normal variables u
        and returns a 1-D array of the gradients nabla g(u)
    u0: 1-D Numpy array giving starting point for optimization
    pf: float giving probability of failure to be searched for
    tfmJac: function which provides transform jacobian; dUdT(u)
    That: covariance matrix for margin estimation

    C: Confidence level for margin computation
    n_restarts: Number of multi-starts for optimization
    tol: Optimizer tolerance
    niter: Maximum iteration count; per-restart

    Returns:
    z: float, limit state level corresponding to pf
    mpp: 1-D array of standard normal variables u giving MPP
    dzdu: 1-D array giving limit state gradient at MPP
    """

    # Target reliability index
    beta = norm.ppf(1 - pf)

    # Margin computation
    def fcn_b(u):
        dgdU  = gradFunc(u)
        dBdT  = np.dot(tfmJac(u).T, dgdU) / np.linalg.norm(dgdU)
        tau2  = np.dot(dBdT, np.dot(That, dBdT))
        b_hat = norm.ppf(C) * np.sqrt(tau2)

        # DEBUG
        # print("b_hat = {0:5.4f}".format(b_hat))

        return b_hat

    # Reliability constraint
    con    = lambda u: u.dot(u) - (beta + fcn_b(u)) ** 2
    # con    = lambda u: u.dot(u) - beta ** 2
    # conJac = lambda u: 2 * u - 2 * fcn_b(u) * fcn_gb(u)
    # conJac = lambda u: 2 * u

    z    = []
    mpp  = []
    dzdu = []

    # Optimize
    result = minimize(func, u0, args = (), method = 'SLSQP', jac = gradFunc,
        tol = tol, options = {'maxiter': niter , 'disp': False},
        constraints=[{'type': 'eq', 'fun': con}])
    # result = minimize(func, u0, args = (), method = 'SLSQP', jac = gradFunc,
    #     tol = tol, options = {'maxiter': niter , 'disp': False},
    #     constraints=[{'type': 'eq', 'fun': con, 'jac': conJac}])

    if result['status'] != 0: # i.e. not success
        print("WARNING: FORM optimization not successful")
        print(result)
        # raise RuntimeError("FORM optimization not successful")
        n_restarts += 1

    # Extract optimization results
    if result['status'] == 0:
        z.append(result.fun)
        mpp.append(result.x)
        dzdu.append(result.jac)

    rg = np.random.RandomState(seed=337)
    for i in range(n_restarts-1):

        u0_2 = rg.uniform(low=0., high=1., size=(len(u0),))
        u0_2 = u0_2*beta/np.linalg.norm(u0_2)

        result = minimize(func, u0_2, args = (), method = 'SLSQP', jac = gradFunc,
            tol = tol, options = {'maxiter': niter, 'disp': False},
            constraints=[{'type': 'eq', 'fun': con}])
        # result = minimize(func, u0_2, args = (), method = 'SLSQP', jac = gradFunc,
        #     tol = tol, options = {'maxiter': niter, 'disp': False},
        #     constraints=[{'type': 'eq', 'fun': con, 'jac': conJac}])

        if result['status'] != 0: # i.e. not success
            print("WARNING: FORM optimization not successful")
            print(result)
            # raise RuntimeError("FORM optimization not successful")

        # Extract optimization results
        if result['status'] == 0:
            z.append(result.fun)
            mpp.append(result.x)
            dzdu.append(result.jac)

    # Choose result that gives lowest limit state value
    ind = np.argmin(z)

    return np.min(z), mpp[ind], dzdu[ind]

if __name__ == '__main__':

    # Short column example
    #########################
    # from column_4v import g_u, dgdu_u

    # x = np.array([0.5,0.5])
    # u0 = np.ones((4,))/np.sqrt(4.)
    # pf = norm.cdf(-3.)

    # func = lambda q: g_u(x, q)
    # gradFunc = lambda q: dgdu_u(x, q)

    # Tension example
    #########################
    # from tension import func_g, grad_g

    # d = 0.027
    # u0 = np.ones(2) / np.sqrt(2)
    # pf = norm.cdf(-1.5)

    # # Scale objectives for stability
    # func     = lambda u: func_g(u, d = d) / 1e8
    # gradFunc = lambda u: grad_g(u, d = d) / 1e8

    # z, mpp, dzdu = pma(func, gradFunc, u0, pf, n_restarts=10)

    # Cantilever beam example
    #########################
    from cantilever import fcn_g_disp, grad_g_disp

    d  = [3.82, 2.49]
    u0 = np.ones(4) / np.sqrt(4)
    pf = norm.cdf(-3)

    ## Scale objectives for stability
    scale = 1e2
    func     = lambda u: fcn_g_disp(u, d) / scale
    gradFunc = lambda u: grad_g_disp(u, d) / scale

    z, mpp, dzdu = pma(func, gradFunc, u0, pf, n_restarts=10)

    g_cr    = z * scale
    dgdu_cr = dzdu * scale

    print(np.linalg.norm(mpp))
