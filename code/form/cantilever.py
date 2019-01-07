### Cantilever beam problem

import numpy as np
from ad import gh
from ad.admath import sqrt, pow
from scipy.stats import norm

## For displacement reliability calculation
from rpy2.robjects.packages import importr
quad = importr("CompQuadForm")

## Script parameters
##################################################
# Nomenclature
# theta = [MU_E, MU_Y, TAU_E, TAU_Y]
# U     = [U_H, U_V, U_E, U_Y]

# Ground truth parameters
MU_H  = 500.
MU_V  = 1000.
MU_E  = 2.9e7
MU_Y  = 40000.

TAU_H = 100.
TAU_V = 100.
TAU_E = 1.45e6
TAU_Y = 2000.

L = 100.
D_MAX = 2.2535

S_STRESS = 1e0
S_DISP   = 1e0

## Function definitions
##################################################
## Objective function
def obj(d):
    return d[0] * d[1]

def objGrad(d):
    return np.array([d[1], d[0]])

## Limit state: Displacement
## --------------------------------------------------
def fcn_g_disp(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    ## Limit state
    return (
        D_MAX**2 / 16. / L**6 * w**6 * t**6 * \
        (mu_e + np.sqrt(tau2_e) * U_E)**2 \
      - w**4 * (MU_V + TAU_V * U_V)**2 \
      - t**4 * (MU_H + TAU_H * U_H)**2
    ) / S_DISP

def grad_g_disp(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    return np.array([
        -2 * t**4 * (MU_H + TAU_H * U_H) * TAU_H,
        -2 * w**4 * (MU_V + TAU_V * U_V) * TAU_V,
         2 * (D_MAX ** 2 / 16 / L ** 6) * w ** 6 * t ** 6 * \
           (mu_e + np.sqrt(tau2_e) * U_E) * np.sqrt(tau2_e),
         0
    ]) / S_DISP

def sens_g_disp(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    g = fcn_g_disp(U = U, d = d, theta = theta)
    return np.array([
        6 * D_MAX**2 / 16. / L**6 * w**5 * t**6 * \
          (mu_e + np.sqrt(tau2_e) * U_E)**2 \
        - 4 * w**3 * (MU_V + TAU_V * U_V)**2,
        6 * D_MAX**2 / 16. / L**6 * w**6 * t**5 * \
          (mu_e + np.sqrt(tau2_e) * U_E)**2 \
        - 4 * t**3 * (MU_H + TAU_H * U_V)**2,
    ]) / S_DISP

def R_disp(d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    w, t = d

    lam = [
        D_MAX**2 * w**6 * t**6 * tau2_e / 16. / L**6,
        - w**4 * TAU_V**2,
        - t**4 * TAU_H**2
    ]
    h   = [1] * len(lam)
    det = [
        mu_e**2 / tau2_e,
        (MU_V / TAU_V)**2,
        (MU_H / TAU_H)**2
    ]

    res = quad.davies(0, lam, h, det)
    return res[-1][0]

## Limit state: Stress
## --------------------------------------------------
def fcn_g_stress(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    ## Limit state
    return (
        (mu_y + np.sqrt(tau2_y) * U_Y) - \
        600 * (MU_V + TAU_V * U_V) / (w * t ** 2) - \
        600 * (MU_H + TAU_H * U_H) / (w ** 2 * t)
    ) / S_STRESS

def grad_g_stress(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    ## Limit state derivative
    return np.array([
        - 600 / w ** 2 / t * TAU_H,
        - 600 / w / t ** 2 * TAU_V,
        0,
        np.sqrt(tau2_y)
    ]) / S_STRESS

def sens_g_stress(U, d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U
    w, t = d

    return np.array([
         600. * np.power(w, -2) * np.power(t, -2) * (MU_V + TAU_V * U_V) + \
        1200. * np.power(w, -3) * np.power(t, -1) * (MU_H + TAU_H * U_H),
        1200. * np.power(w, -1) * np.power(t, -3) * (MU_V + TAU_V * U_V) + \
         600. * np.power(w, -2) * np.power(t, -2) * (MU_H + TAU_H * U_H)
    ]) / S_STRESS

## Exact reliability analysis
def zc_stress(d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    w, t = d

    zc = (mu_y - 600 * MU_V / w / t ** 2 - 600 * MU_H / w ** 2 / t) / \
         np.sqrt(
             600 ** 2 / w ** 2 / t ** 4 * TAU_V ** 2 + \
             600 ** 2 / w ** 4 / t ** 2 * TAU_H ** 2 + \
             tau2_y
         )

    return zc

def R_stress(d, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    return norm.cdf(zc_stress(d, theta = theta))

## Random variable parameterization
## --------------------------------------------------
def dUdT(U, theta = [MU_E, MU_Y, TAU_E ** 2, TAU_Y ** 2]):
    ## Renaming
    mu_e  = theta[0]
    mu_y  = theta[1]
    tau2_e = theta[2]
    tau2_y = theta[3]

    U_H, U_V, U_E, U_Y = U

    return np.array([
        [0                   , 0                   , 0                  , 0],
        [0                   , 0                   , 0                  , 0],
        [-1 / np.sqrt(tau2_e), 0                   , -0.5 * U_E / tau2_e, 0],
        [0                   , -1 / np.sqrt(tau2_y), 0                  , -0.5 * U_Y / tau2_y]
    ])

def gen_That(theta, m):
    return np.diag([
        theta[2] / m,
        theta[3] / m,
        2 * theta[2] ** 2 / (m - 1),
        2 * theta[3] ** 2 / (m - 1)
    ])

if __name__ == "__main__":
    from form import pma

    test_reliability = False
    test_gradients   = True
    test_opt         = False

    ## Test reliability analysis
    ##################################################
    if test_reliability:
        print("Reliability Calculation")
        print("--------------------------------------------------")
        N   = int(1e5)
        m   = 1e18 # Assume m_e = m_y
        Con = 0.9

        U = np.random.normal(size = (4, N))
        That = np.diag([
            TAU_E ** 2 / m,
            TAU_Y ** 2 / m,
            2 * TAU_E ** 4 / (m - 1),
            2 * TAU_Y ** 4 / (m - 1)
        ])
        u0 = np.ones(4) / np.sqrt(4)

        ## Stress
        ## --------------------------------------------------
        d0 = np.array([2.6999, 3.4098])
        ## Exact reliability analysis
        pf_stress_ext = 1 - R_stress(d0)
        print("pf_stress_ext = {}".format(pf_stress_ext))

        ## MCS
        G_stress = fcn_g_stress(U, d = d0)
        pf_stress_hat = np.mean(G_stress <= 0)
        print("pf_stress_hat = {}".format(pf_stress_hat))

        ## PMA
        tfm = lambda u: dUdT(u)
        fcn_stress = lambda u: fcn_g_stress(u, d = d0) / 1e4
        grd_stress = lambda u: grad_g_stress(u, d = d0) / 1e4

        g_cr_stress, mpp_stress, dgdU_stress = pma(
            fcn_stress,
            grd_stress,
            u0,
            pf_stress_ext,
            tfm,
            That
        )

        print("g_stress    = {0:5.4f}".format(g_cr_stress))
        # print("mpp  = {}".format(mpp))
        # print("dgdU = {}".format(dgdU))

        ## Displacement
        ## --------------------------------------------------
        d0 = np.array([2.448387, 3.888371])
        ## Analysis via Davies algorithm
        pf_disp_ext = 1 - R_disp(d0)
        print("pf_disp_ext = {}".format(pf_disp_ext))

        ## MCS
        G_disp = fcn_g_disp(U, d = d0)
        pf_disp_hat = np.mean(G_disp <= 0)
        print("pf_disp_hat = {}".format(pf_disp_hat))

        ## PMA
        fcn_disp = lambda u: fcn_g_disp(u, d = d0) / 1e8
        grd_disp = lambda u: grad_g_disp(u, d = d0) / 1e8

        g_cr_disp, mpp_disp, dgdU_disp = pma(
            fcn_disp,
            grd_disp,
            u0,
            pf_disp_ext,
            tfm,
            That
        )

        print("g_disp    = {0:5.4f}".format(g_cr_disp))

    ## Verify gradients
    ##################################################
    if test_gradients:
        print("")
        print("Gradients")
        print("--------------------------------------------------")
        import pyutil.numeric as ut

        d0 = np.array([2.446077, 3.892212])
        u0 = np.ones(4) / np.sqrt(4)
        ## Objective function gradient
        # --------------------------------------------------
        obj_noise, _, info = ut.autonoise(
            fcn = lambda t: obj(d0 + np.ones(2) * t),
            t0  = 0
        )
        obj_h, _ = ut.stepest(
            fcn   = lambda t: obj(d0 + np.ones(2) * t),
            t0    = 0,
            eps_f = obj_noise
        )

        print("obj_noise = {0:4.3e}".format(obj_noise))
        print("obj_h     = {0:4.3e}".format(obj_h))

        obj_grad_fd = ut.grad(d0, obj, h = obj_h)
        obj_grad_an = objGrad(d0)
        obj_grad_diff = obj_grad_an - obj_grad_fd
        obj_grad_rel  = obj_grad_diff / np.linalg.norm(obj_grad_an)

        print("obj_grad_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(obj_grad_diff),
            np.linalg.norm(obj_grad_rel),
            obj_grad_diff
        ))
        print()
        print("obj_grad_an = {}".format(obj_grad_an))

        ## Limit state Displacement: gradient
        # --------------------------------------------------
        print("")
        g_disp_noise, _, info_noise_g_disp = ut.autonoise(
            fcn = lambda t: fcn_g_disp(u0 + np.ones(4) * t, d = d0),
            t0  = 0,
            h0  = 1e-6
        )
        g_disp_h, info_step_g_disp = ut.stepest(
            fcn = lambda t: fcn_g_disp(u0 + np.ones(4) * t, d = d0),
            t0    = 0,
            eps_f = g_disp_noise
        )

        print("g_disp_noise = {0:4.3e}".format(g_disp_noise))
        print("g_disp_h     = {0:4.3e}".format(g_disp_h))

        g_disp_grad_fd = ut.grad(
            x = u0,
            f = lambda u: fcn_g_disp(u, d = d0),
            h = g_disp_h
        )
        g_disp_grad_an = grad_g_disp(u0, d = d0)
        g_disp_grad_diff = g_disp_grad_an - g_disp_grad_fd
        g_disp_grad_rel  = g_disp_grad_diff / np.linalg.norm(g_disp_grad_an)

        print("g_disp_grad_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(g_disp_grad_diff),
            np.linalg.norm(g_disp_grad_rel),
            g_disp_grad_diff
        ))
        print()
        print("g_disp_grad_an = {}".format(g_disp_grad_an))

        ## Limit state Stress: gradient
        # --------------------------------------------------
        print("")
        g_stress_noise, _, info_noise_g_stress = ut.autonoise(
            fcn = lambda t: fcn_g_stress(u0 + np.ones(4) * t, d = d0),
            t0  = 0,
            h0  = 1e-4
        )
        g_stress_h, info_step_g_stress = ut.stepest(
            fcn = lambda t: fcn_g_stress(u0 + np.ones(4) * t, d = d0),
            t0    = 0,
            eps_f = g_stress_noise
        )

        print("g_stress_noise = {0:4.3e}".format(g_stress_noise))
        print("g_stress_h     = {0:4.3e}".format(g_stress_h))

        g_stress_grad_fd = ut.grad(
            x = u0,
            f = lambda u: fcn_g_stress(u, d = d0),
            h = g_stress_h
        )
        g_stress_grad_an = grad_g_stress(u0, d = d0)
        g_stress_grad_diff = g_stress_grad_an - g_stress_grad_fd
        g_stress_grad_rel  = g_stress_grad_diff / np.linalg.norm(g_stress_grad_an)

        print("g_stress_grad_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(g_stress_grad_diff),
            np.linalg.norm(g_stress_grad_rel),
            g_stress_grad_diff
        ))
        print()
        print("g_stress_grad_an = {}".format(g_stress_grad_an))

        ## Limit state Displacement: sensitivity
        # --------------------------------------------------
        print("")
        s_disp_noise, _, info_noise_s_disp = ut.autonoise(
            fcn = lambda t: fcn_g_disp(u0, d = d0 + np.ones(2) * t),
            t0  = 0,
            # h0  = 1e-6
        )
        s_disp_h, info_step_s_disp = ut.stepest(
            fcn = lambda t: fcn_g_disp(u0, d = d0 + np.ones(2) * t),
            t0    = 0,
            eps_f = s_disp_noise
        )

        print("s_disp_noise = {0:4.3e}".format(s_disp_noise))
        print("s_disp_h     = {0:4.3e}".format(s_disp_h))

        g_disp_sens_fd = ut.grad(
            x = d0,
            f = lambda d: fcn_g_disp(u0, d = d),
            h = s_disp_h
        )
        g_disp_sens_an = sens_g_disp(u0, d0)
        g_disp_sens_diff = g_disp_sens_an - g_disp_sens_fd
        g_disp_sens_rel  = g_disp_sens_diff / np.linalg.norm(g_disp_sens_an)

        print("g_disp_sens_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(g_disp_sens_diff),
            np.linalg.norm(g_disp_sens_rel),
            g_disp_sens_diff
        ))
        print()
        print("g_disp_sens_an = {}".format(g_disp_sens_an))

        ## Limit state Stress: sensitivity
        # --------------------------------------------------
        print("")
        s_stress_noise, _, info_noise_s_stress = ut.autonoise(
            fcn = lambda t: fcn_g_stress(u0, d = d0 + np.ones(2) * t),
            t0  = 0,
            # h0  = 1e-6
        )
        s_stress_h, info_step_s_stress = ut.stepest(
            fcn = lambda t: fcn_g_stress(u0, d = d0 + np.ones(2) * t),
            t0    = 0,
            eps_f = s_stress_noise
        )

        print("s_stress_noise = {0:4.3e}".format(s_stress_noise))
        print("s_stress_h     = {0:4.3e}".format(s_stress_h))

        g_stress_sens_fd = ut.grad(
            x = d0,
            f = lambda d: fcn_g_stress(u0, d = d),
            h = s_stress_h
        )
        g_stress_sens_an = sens_g_stress(u0, d0)
        g_stress_sens_diff = g_stress_sens_an - g_stress_sens_fd
        g_stress_sens_rel  = g_stress_sens_diff / np.linalg.norm(g_stress_sens_an)

        print("g_stress_sens_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(g_stress_sens_diff),
            np.linalg.norm(g_stress_sens_rel),
            g_stress_sens_diff
        ))
        print()
        print("g_stress_sens_an = {}".format(g_stress_sens_an))

        ## FORM gradients
        # --------------------------------------------------
        print("")
        scale = 1
        def run_FORM(d):
            fcn = lambda u: fcn_g_stress(u, d = d) / scale
            grd = lambda u: grad_g_stress(u, d = d) / scale
            tfm = lambda u: dUdT(u)
            u0 = np.ones(4) / np.sqrt(4)

            g_cr, mpp, dgdU = pma(
                func     = fcn,
                gradFunc = grd,
                u0       = u0,
                pf       = 0.1,
                tfmJac   = tfm,
                That     = np.diag([0.] * 4)
            )

            return g_cr, mpp

        gc_stress_noise, _, info_noise_gc_stress = ut.autonoise(
            fcn = lambda t: run_FORM(d0 + np.ones(2) * t)[0],
            t0  = 0,
            # h0  = 1e-6
        )
        gc_stress_h, info_step_gc_stress = ut.stepest(
            fcn = lambda t: run_FORM(d0 + np.ones(2) * t)[0],
            t0    = 0,
            eps_f = gc_stress_noise
        )

        print("gc_stress_noise = {0:4.3e}".format(gc_stress_noise))
        print("gc_stress_h     = {0:4.3e}".format(gc_stress_h))

        gc_stress_sens_fd = ut.grad(
            x = d0,
            f = lambda d: run_FORM(d)[0],
            h = gc_stress_h
        )
        gc_mpp = run_FORM(d0)[1]
        gc_stress_sens_an   = sens_g_stress(gc_mpp, d0)
        gc_stress_sens_diff = gc_stress_sens_an - gc_stress_sens_fd
        gc_stress_sens_rel  = gc_stress_sens_diff / np.linalg.norm(gc_stress_sens_an)

        print("gc_stress_sens_diff: abs: {0:4.3e} rel: {1:4.3e}\n {2:}".format(
            np.linalg.norm(gc_stress_sens_diff),
            np.linalg.norm(gc_stress_sens_rel),
            gc_stress_sens_diff
        ))

    ## Test optimization
    ##################################################
    if test_opt:

        d0 = np.array([3.0, 3.0])
        u0 = np.ones(4) / np.sqrt(4)

        m = 1e3
        theta = np.array([MU_E, MU_Y, TAU_E, TAU_Y])
        # That = gen_That(theta, m)
        That = np.diag([0] * 4)

        pf_stress_target = 0.00135
        pf_disp_target   = 0.00135

        import pyOpt

        def objfunc(x):
            # f = objective value
            # g = [-gc_stress, -gc_disp]
            f = obj(x)
            g = [0] * 2
            try:
                gc_stress, _, _ = pma(
                    func     = lambda u: fcn_g_stress(u, d = x),
                    gradFunc = lambda u: grad_g_stress(u, d = x),
                    u0       = u0,
                    pf       = pf_stress_target,
                    tfmJac   = lambda u: dUdT(u),
                    That     = That
                )
                g[0]  = -gc_stress

                gc_disp, _, _ = pma(
                    func     = lambda u: fcn_g_disp(u, d = x),
                    gradFunc = lambda u: grad_g_disp(u, d = x),
                    u0       = u0,
                    pf       = pf_disp_target,
                    tfmJac   = lambda u: dUdT(u),
                    That     = That
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
                    func     = lambda u: fcn_g_stress(u, d = x),
                    gradFunc = lambda u: grad_g_stress(u, d = x),
                    u0       = u0,
                    pf       = pf_stress_target,
                    tfmJac   = lambda u: dUdT(u),
                    That     = That
                )
                grad_con[0] = -sens_g_stress(U = mpp_stress, d = x)

                _, mpp_disp, _ = pma(
                    func     = lambda u: fcn_g_disp(u, d = x),
                    gradFunc = lambda u: grad_g_disp(u, d = x),
                    u0       = u0,
                    pf       = pf_disp_target,
                    tfmJac   = lambda u: dUdT(u),
                    That     = That
                )
                grad_con[1] = -sens_g_disp(U = mpp_disp, d = x)

                fail = 0
            except ValueError:
                fail = 1

            return grad_obj, grad_con, fail

        opt_prob = pyOpt.Optimization("Cantilever Beam", objfunc)
        opt_prob.addObj("f")
        opt_prob.addVar("x1", "c", lower = 1.0, upper = 4.0, value = 3.0)
        opt_prob.addVar("x2", "c", lower = 1.0, upper = 4.0, value = 3.0)
        opt_prob.addCon("g1", "i")
        opt_prob.addCon("g2", "i")

        print(opt_prob)

        slsqp = pyOpt.SLSQP()
        slsqp.setOption("IPRINT", -1)
        [fstr, xstr, inform] = slsqp(opt_prob, sens_type = gradfunc)

        print(opt_prob.solution(0))

        ds = [0] * 2
        ds[0] = opt_prob.solution(0)._variables[0].value
        ds[1] = opt_prob.solution(0)._variables[1].value

        fs = opt_prob.solution(0)._objectives[0].value
