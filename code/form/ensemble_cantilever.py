import numpy as np
import time
import pandas as pd
import pyOpt

import pyutil.numeric as ut

from cantilever import obj, objGrad
from cantilever import fcn_g_stress, grad_g_stress, sens_g_stress, R_stress
from cantilever import fcn_g_disp, grad_g_disp, sens_g_disp, R_disp
from cantilever import dUdT, gen_That, MU_E, MU_Y, TAU_E, TAU_Y

from itertools import chain
from form import pma
from scipy.stats import norm, chi2

np.random.seed(101)

## Script parameters
##################################################
# M_ALL = [10, 20, 50, int(1e2), int(1e3), int(1e4)]; repl  = 300; Con = 0.80; mycase = 0
# M_ALL = [10, 20, 50, int(1e2), int(1e3), int(1e4)]; repl  = 300; Con = 0.90; mycase = 1
M_ALL = [50, 60, 70, 80, 90, 100]; repl  = 300; Con = 0.80; mycase = 2

print("mycase = {}".format(mycase))

scale     = 1e8
tol       = 1e-3
niter     = int(10)
silent    = True

pf_stress = 0.00135
pf_disp   = 0.00135

## Define optimization problem
##################################################
d0        = np.array([3., 3.])
u0        = np.ones(4) / np.sqrt(4)
THETA     = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]

def opt_run(theta, m):
    ## Run a parameterized optimization problem
    # Usage
    #   fs, ds = opt_run(theta, m)
    # Arguments
    #   theta = estimated parameter vector
    #   m     = sample count
    # Returns
    #   fs    = optimum value
    #   ds    = optimum point

    That = gen_That(theta, m)

    def objfunc(x):
        # f = objective value
        # g = [-gc_stress, -gc_disp]
        f = obj(x)
        g = [0] * 2
        try:
            gc_stress, _, _ = pma(
                func     = lambda u: fcn_g_stress(u, d = x, theta = theta),
                gradFunc = lambda u: grad_g_stress(u, d = x, theta = theta),
                u0       = u0,
                pf       = pf_stress,
                tfmJac   = lambda u: dUdT(u, theta = theta),
                That     = That,
                C        = Con
            )
            g[0]  = -gc_stress

            gc_disp, _, _ = pma(
                func     = lambda u: fcn_g_disp(u, d = x, theta = theta),
                gradFunc = lambda u: grad_g_disp(u, d = x, theta = theta),
                u0       = u0,
                pf       = pf_disp,
                tfmJac   = lambda u: dUdT(u, theta = theta),
                That     = That,
                C        = Con
            )
            g[1]  = -gc_disp

            fail = 0
        except ValueError:
            fail = 1

        return f, g, fail

    def gradfunc(x, f, g):
        grad_obj = [0] * 2
        grad_obj[:] = objGrad(x)
        grad_con = np.zeros((2, 2))
        try:
            _, mpp_stress, _ = pma(
                func     = lambda u: fcn_g_stress(u, d = x, theta = theta),
                gradFunc = lambda u: grad_g_stress(u, d = x, theta = theta),
                u0       = u0,
                pf       = pf_stress,
                tfmJac   = lambda u: dUdT(u, theta = theta),
                That     = That,
                C        = Con
            )
            grad_con[0] = -sens_g_stress(U = mpp_stress, d = x, theta = theta)

            _, mpp_disp, _ = pma(
                func     = lambda u: fcn_g_disp(u, d = x, theta = theta),
                gradFunc = lambda u: grad_g_disp(u, d = x, theta = theta),
                u0       = u0,
                pf       = pf_disp,
                tfmJac   = lambda u: dUdT(u, theta = theta),
                That     = That,
                C        = Con
            )
            grad_con[1] = -sens_g_disp(U = mpp_disp, d = x, theta = theta)

            fail = 0
        except ValueError:
            fail = 1

        return grad_obj, grad_con, fail

    opt_prob = pyOpt.Optimization("Cantilever Beam", objfunc)
    opt_prob.addObj("f")
    opt_prob.addVar("x1", "c", lower = 2.0, upper = 4.0, value = 3.0)
    opt_prob.addVar("x2", "c", lower = 2.0, upper = 4.0, value = 3.0)
    opt_prob.addCon("g1", "i")
    opt_prob.addCon("g2", "i")

    slsqp = pyOpt.SLSQP()
    slsqp.setOption("IPRINT", -1)
    [fstr, xstr, inform] = slsqp(opt_prob, sens_type = gradfunc)

    ds = [0] * 2
    ds[0] = opt_prob.solution(0)._variables[0].value
    ds[1] = opt_prob.solution(0)._variables[1].value

    fs = opt_prob.solution(0)._objectives[0].value

    return fs, ds

## Ensemble of optimization runs
##################################################
## Simulate parameter sampling
Theta = np.zeros((repl, len(M_ALL), 4))
for ind in range(len(M_ALL)):
    Theta[:, ind, 0] = np.random.normal(
        loc = MU_E,
        scale = TAU_E / np.sqrt(M_ALL[ind]),
        size = repl
    )
    Theta[:, ind, 1] = np.random.normal(
        loc = MU_Y,
        scale = TAU_Y / np.sqrt(M_ALL[ind]),
        size = repl
    )
    Theta[:, ind, 2] = chi2.rvs(
        df = M_ALL[ind] - 1,
        size = repl
    ) * TAU_E ** 2 / (M_ALL[ind] - 1)
    Theta[:, ind, 3] = chi2.rvs(
        df = M_ALL[ind] - 1,
        size = repl
    ) * TAU_Y ** 2 / (M_ALL[ind] - 1)

## Reserve space
D_ALL = np.zeros((repl, len(M_ALL), 2))
R_ALL_stress = np.zeros((repl, len(M_ALL)))
R_ALL_disp = np.zeros((repl, len(M_ALL)))
G_ALL = np.zeros((repl, len(M_ALL)))

## Main loop
# --------------------------------------------------
## Infinite-sample 'true' solution
fs_true, ds_true = opt_run(THETA, 1e32)

t0 = time.time()
ut.print_progress(0, repl - 1, bar_length = 60)
for ind in range(repl):
    for jnd in range(len(M_ALL)):
        theta   = Theta[ind, jnd]

        fs, ds = opt_run(theta, M_ALL[jnd])

        D_ALL[ind, jnd]        = ds
        R_ALL_stress[ind, jnd] = R_stress(D_ALL[ind, jnd])
        R_ALL_disp[ind, jnd]   = R_disp(D_ALL[ind, jnd])
        G_ALL[ind, jnd]        = (fs - fs_true) / fs_true

    ut.print_progress(ind, repl - 1, bar_length = 60)
t1 = time.time()

## Post-process
##################################################
C_ALL = np.mean(G_ALL >= 0, axis = 0)

D_mu = np.mean(D_ALL, axis = 0)
D_lo = D_ALL[int(repl * (1 - Con)), :]
D_hi = D_ALL[int(repl * Con), :]

G_ALL.sort(axis = 0)
G_mu = np.mean(G_ALL, axis = 0)
G_lo = G_ALL[int(repl * (1 - Con)), :]
G_hi = G_ALL[int(repl * Con), :]

R_mu_stress = np.mean(R_ALL_stress, axis = 0)
R_lo_stress = R_ALL_stress[int(repl * (1 - Con)), :]
R_hi_stress = R_ALL_stress[int(repl * Con), :]
R_mu_disp = np.mean(R_ALL_disp, axis = 0)
R_lo_disp = R_ALL_disp[int(repl * (1 - Con)), :]
R_hi_disp = R_ALL_disp[int(repl * Con), :]

## Summary data
df = pd.DataFrame(
    data = {
        "N"        : M_ALL,
        "R_stress" : [1 - pf_stress] * len(M_ALL),
        "R_disp"   : [1 - pf_disp] * len(M_ALL),
        "C"        : [Con] * len(M_ALL),

        "d1_mu" : D_mu[:, 0],
        "d2_mu" : D_mu[:, 1],
        "d1_lo" : D_lo[:, 0],
        "d2_lo" : D_lo[:, 1],
        "d1_hi" : D_hi[:, 0],
        "d2_hi" : D_hi[:, 1],

        "M_mu" : G_mu,
        "M_lo" : G_lo,
        "M_hi" : G_hi,

        "R_mu_stress" : R_mu_stress,
        "R_hi_stress" : R_hi_stress,
        "R_lo_stress" : R_lo_stress,
        "R_mu_disp"   : R_mu_disp,
        "R_hi_disp"   : R_hi_disp,
        "R_lo_disp"   : R_lo_disp
    }
)
df.to_csv("../../data/ensemble_cantilever_c{}.csv".format(mycase))

## Design ensemble
data_d1 = {
    "d1_N{0:}".format(M_ALL[j]): D_ALL[:, j, 0]
    for j in range(len(M_ALL))
}
data_d2 = {
    "d2_N{0:}".format(M_ALL[j]): D_ALL[:, j, 1]
    for j in range(len(M_ALL))
}

df_designs = pd.DataFrame(
    data = dict(chain(data_d1.items(), data_d2.items()))
)
df_designs.to_csv("../../data/designs_cantilever_c{}.csv".format(mycase))

## Report
##################################################
print("Execution time: {0:3.2f} sec".format(t1 - t0))
print("({})".format(Con))
print("C_ALL = {}".format(C_ALL))
print("")
print("G_mu = {}".format(G_mu))
print("G_lo = {}".format(G_lo))
print("")
print("R_mu_stress = {}".format(R_mu_stress))
print("R_lo_stress = {}".format(R_lo_stress))
print("R_mu_disp   = {}".format(R_mu_disp))
print("R_lo_disp   = {}".format(R_lo_disp))
