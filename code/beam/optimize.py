"""
Subroutines for optimization of cantilevered beam problem.

Zach del Rosario and Richard W. Fenrich, April 24, 2018
"""

import numpy as np 
from scipy.stats import norm, nct 
from scipy.optimize import minimize
import collections
import pyOpt 

from beam import area, stress, disp
from beam import areaJac, stressJac, dispJac
from beam import deltaDFactor, covarianceDistributionParams

np.random.seed(295)
np.set_printoptions(precision=6)

def calcMIPLDMResponse(d, x, info, func=None):
    """ Calculate F^{-1}(p_f) for limit state with margin in probability in
    limit dispersion margin included. """
    g = func(d, x)
    factor = info["factor"] # \frac{\nabla \rho}{\rho}
    ig = g > 0
    delR = (1./info["N"])*ig.dot(factor)
    Tm = info["Tm"]
    varR_s = delR.dot(Tm.dot(delR))
    coef = info["coef"]
    pc = coef*np.sqrt(varR_s) # margin in probability
    con = calcResponse(d, x, info["pf"], info["N"], func=func, 
        probability_margin=pc)
    return con

def calcMDLDMResponse(d, x, info, func=None):
    """ Calculate F^{-1}(p_f) for limit state with mean difference limit 
    dispersion margin included. """
    g = func(d, x)
    factor = info["factor"] # \frac{\nabla \rho}{\rho}
    delD = np.mean(factor.T*g, axis=1)
    Tm = info["Tm"]
    varD_s = delD.dot(Tm.dot(delD))
    coef = info["coef"]
    md_ldm = coef*np.sqrt(varD_s) # mean difference limit dispersion margin
    con = calcResponse(d, x, info["pf"], info["N"], func=func, 
        limit_margin=md_ldm)
    return con

def calcMDLDMJacobian(d, x, info, func=None, jac=None):
    """ Calculate Jacobian for F^{-1}(p_f) for limit state with mean 
    difference limit dispersion margin included. """
    # g = func(d, x)
    # factor = info["factor"] # \frac{\nabla \rho}{\rho}
    # delD = np.mean(factor.T*g, axis=1)
    # Tm = info["Tm"]
    # varD_s = delD.dot(Tm.dot(delD))
    # coef = info["coef"]
    # # Jacobian of limit state with no margin
    # gJac = calcJacobian(d, x, info["pf"], info["N"], func=func, jac=jac)
    # gJacAll = jac(d, x) # Jacobian for all samples
    # # Jacobian of margin
    # Gprime = coef/(2*np.sqrt(varD_s))
    # T2prime = 2*Tm.dot(delD)
    # delDprime = (1./info["N"])*factor.T.dot(gJacAll)
    # limJac = Gprime*T2prime.dot(delDprime)
    # # print gJac, limJac
    # # Total Jacobian
    # conJac = gJac - limJac
    # # print(conJac)

    # Derivative estimated using finite difference
    # XXX Will need to test with larger margin so margin derivative is bigger
    delta = 1e-8
    d1 = np.copy(d) + np.array([delta, 0])
    d2 = np.copy(d) + np.array([0, delta])
    con = calcMDLDMResponse(d, x, info, func=func)
    con1 = calcMDLDMResponse(d1, x, info, func=func)
    con2 = calcMDLDMResponse(d2, x, info, func=func)
    conJac = (np.array([con1, con2]) - con)/delta
    # print("Jacobian from finite difference:")
    # print(conJac)
    # sys.exit()
        
    return conJac

def calcMIPLDMJacobian(d, x, info, func=None, jac=None):
    """ Calculate Jacobian for F^{-1}(p_f) for limit state with margin in
    probability limit dispersion margin included. """
    # Derivative estimated using finite difference
    delta = 1e-8
    d1 = np.copy(d) + np.array([delta, 0])
    d2 = np.copy(d) + np.array([0, delta])
    con = calcMIPLDMResponse(d, x, info, func=func)
    con1 = calcMIPLDMResponse(d1, x, info, func=func)
    con2 = calcMIPLDMResponse(d2, x, info, func=func)
    conJac = (np.array([con1, con2]) - con)/delta 
    return conJac   
    
def calcResponse(d, x, pf, N, func=None, limit_margin=None, probability_margin=None):
    """ Calculate F^{-1}(p_f), i.e. inverse CDF evaluated at p_f. """
    g = func(d, x)
    gsort = np.sort(g)
    psort = np.linspace(1./N,1.,N)
    if limit_margin is not None:
        con = np.interp(pf, psort, gsort-limit_margin)
    elif probability_margin is not None:
        pf = np.max((1./N, pf-probability_margin))
        con = np.interp(pf, psort, gsort)
    else:
        con = np.interp(pf, psort, gsort) # constraint value at pf
    return con

def calcJacobian(d, x, pf, N, func=None, jac=None):
    """ Calculate Jacobian for F^{-1}(p_f), i.e. Jacobian of inverse CDF. """

    # Data common to all derivative approaches
    g = func(d, x)

    # Data common to finite difference approach
    gsort = np.sort(g)
    psort = np.linspace(1./N,1.,N)
    con = np.interp(pf, psort, gsort) # constraint value at pf

    # Data common to chain rule approach
    isort = np.argsort(g)
    #iR = np.argmax(psort>pf) # index of first entry in psort that has prob > pf
    iR = int(pf*N)
    xcon = x[isort[iR],:] # random variable values corresponding to constraint value at approx. pf

    # Derivative estimated using chain rule
    conJac = jac(d, xcon)
    # print("Jacobian from chain rule:")
    # print(conJac)

    # Derivative estimated using finite difference
    delta = 1e-8
    d1 = np.copy(d) + np.array([delta, 0])
    d2 = np.copy(d) + np.array([0, delta])
    g1 = func(d1, x)
    g2 = func(d2, x)
    gsort1 = np.sort(g1)
    gsort2 = np.sort(g2)
    con1 = np.interp(pf, psort, gsort1)
    con2 = np.interp(pf, psort, gsort2)
    conJac = (np.array([con1, con2]) - con)/delta
    # print("Jacobian from finite difference:")
    # print(conJac)

    return conJac

def obj(d, X):
    obj = area(d, X) 
    return obj

def objJac(d, X):
    Jobj = areaJac(d, X)
    return Jobj

def sCon(d, x, info):
    """ Given d and x calculate stress constraint for desired formulation."""
    formulation = info["formulation"]
    if formulation is "det":
        con = stress(d, x)
    elif formulation is "pi": # plug-in
        con = calcResponse(d, x, info["pf"], info["N"], func=stress)
    elif formulation is "bv": # basis value
        con = calcResponse(d, x, info["pf"], info["N"], func=stress)
    elif formulation is "md_ldm": # mean difference limit dispersion margin
        con = calcMDLDMResponse(d, x, info, func=stress)
    elif formulation is "mip_ldm": # probability in margin limit dispersion margin
        con = calcMIPLDMResponse(d, x, info, func=stress)
    else:
        raise NotImplementedError("Formulation %s is not implemented." % formulation)
    return con 

def sConJac(d, x, info):
    """ Given d and x calculate stress constraint Jacobian for desired 
    formulation."""
    formulation = info["formulation"]
    if formulation is "det":
        conJac = stressJac(d, x)
    elif formulation is "pi": # plug-in
        conJac = calcJacobian(d, x, info["pf"], info["N"], func=stress, jac=stressJac) 
    elif formulation is "bv": # basis value
        conJac = calcJacobian(d, x, info["pf"], info["N"], func=stress, jac=stressJac) 
    elif formulation is "md_ldm": # mean difference limit dispersion margin
        conJac = calcMDLDMJacobian(d, x, info, func=stress, jac=stressJac)
    elif formulation is "mip_ldm": # probability in margin limit dispersion margin
        conJac = calcMIPLDMJacobian(d, x, info, func=stress, jac=stressJac)
    else:
        raise NotImplementedError("Formulation %s is not implemented." % formulation)
    return conJac

def dCon(d, x, info):
    """ Given d and x calculate displacement constraint for desired formulation."""
    formulation = info["formulation"]
    if formulation is "det":
        con = disp(d, x)
    elif formulation is "pi": # plug-in
        con = calcResponse(d, x, info["pf"], info["N"], func=disp)
    elif formulation is "bv": # basis value
        con = calcResponse(d, x, info["pf"], info["N"], func=disp)
    elif formulation is "md_ldm": # mean difference limit dispersion margin
        con = calcMDLDMResponse(d, x, info, func=disp)
    elif formulation is "mip_ldm": # probability in margin limit dispersion margin
        con = calcMIPLDMResponse(d, x, info, func=disp)
    else:
        raise NotImplementedError("Formulation %s is not implemented." % formulation)
    return con 

def dConJac(d, x, info):
    """ Given d and x calculate displacement constraint Jacobian for desired 
    formulation."""
    formulation = info["formulation"]
    if formulation is "det":
        conJac = dispJac(d, x)
    elif formulation is "pi": # plug-in
        conJac = calcJacobian(d, x, info["pf"], info["N"], func=disp, jac=dispJac) 
    elif formulation is "bv": # basis value
        conJac = calcJacobian(d, x, info["pf"], info["N"], func=disp, jac=dispJac) 
    elif formulation is "md_ldm": # mean difference limit dispersion margin
        conJac = calcMDLDMJacobian(d, x, info, func=disp, jac=dispJac)
    elif formulation is "mip_ldm": # probability in margin limit dispersion margin
        conJac = calcMIPLDMJacobian(d, x, info, func=disp, jac=dispJac)
    else:
        raise NotImplementedError("Formulation %s is not implemented." % formulation)
    return conJac

def pyopt_func(d, **kwargs):
    """ Function which returns objective and constraint values for pyOpt. """
    X = kwargs['X']
    info = kwargs['info']
    f = obj(d, X)
    gs = sCon(d, X, info)
    gd = dCon(d, X, info)
    g = np.hstack((gs,gd))
    fail = 0
    return f, g, fail

def pyopt_gradfunc(d, f, g, **kwargs):
    """ Function which returns gradients of objective and constraints for pyOpt. """
    X = kwargs['X']
    info = kwargs['info']
    df = np.array([objJac(d, X)])
    dgs = sConJac(d, X, info)
    dgd = dConJac(d, X, info)
    dg = np.vstack((dgs, dgd))
    fail = 0
    return df, dg, fail

def fK_pc(p, c, n):
    """ Basis value knockdown factor for normally distributed random variables."""
    return nct.ppf(c,n-1,-norm.ppf(1-p)*np.sqrt(n)) / np.sqrt(n)

def optimize(d0, X_mean, X_stddev, m_samples, formulation="det", info={},
    tol=1e-6, maxiter=100, bounds=[(1,4),(1,4)], random_number_generator=None):
    """
    Call optimization.
    
    Arguments:
    d0: initial deterministic design variable values = [t, w]
    X_mean: list of length 4 containing estimated mean of random variables:
        [H, V, E, S] 
        H = lateral tip load
        V = vertical tip load
        E = Young's modulus of beam
        S = yield strength of beam        
    X_stddev: list of length 4 containing estimated standard deviation of 
        random variables
    m_samples: number of samples that were used to estimate X_mean and X_stddev
    formulation: "det", "bv", "md_ldm", or "mip_ldm" denoting which constraint
        formulation to use. All chance constraints use the Performance Measure
        Approach.
    """

    f = formulation
    info["formulation"] = f 

    # Get samples to evaluate probabilities with
    if f == "det": # deterministic at mean values
        x = X_mean
    else:
        if random_number_generator is not None:
            x = random_number_generator.multivariate_normal(mean=X_mean,
                                    cov=np.diag(X_stddev**2),
                                    size=info["N"])
        elif type(random_number_generator) is np.ndarray():
            x = random_number_generator
        else:
            x = np.random.multivariate_normal(mean=X_mean,
                                    cov=np.diag(X_stddev**2),
                                    size=info["N"])

    # Additional data for mean difference limit dispersion margin
    if formulation == "md_ldm" or formulation == "mip_ldm":
        info["mean"] = list(X_mean)
        info["stddev"] = list(X_stddev)
        info["factor"] = deltaDFactor(x, info["mean"], info["stddev"])
        info["Tm"] = covarianceDistributionParams(X_mean, X_stddev, m_samples)
        info["coef"] = norm.ppf(info["ldm_confidence_interval"])

    # Setup problem
    if f == "det":
        pass
    elif f == "pi": # plug-in 
        pass
    elif f == "bv": # plug-in for loads, basis values for material properties
        ke = fK_pc(info["bv_p"], info["bv_c"], m_samples)
        ks = fK_pc(info["bv_p"], info["bv_c"], m_samples)
        E_bv = X_mean[2] - ke*X_stddev[2]
        S_bv = X_mean[3] - ks*X_stddev[3]
        x[:,2] = E_bv
        x[:,3] = S_bv
    elif f == "md_ldm": # mean difference limit dispersion margin
        pass
    elif f == "mip_ldm": # probability in margin limit dispersion margin
        pass
    else:
        raise NotImplementedError("Formulation %s is not implemented." % f)
   
    # Run optimization
    # opt_prob = pyOpt.Optimization('Cantilever_beam',pyopt_func)
    # opt_prob.addVar('t', 'c', value=d0[0], lower=bounds[0][0], upper=bounds[0][1])
    # opt_prob.addVar('w', 'c', value=d0[1], lower=bounds[1][0], upper=bounds[1][1])
    # opt_prob.addObj('f')
    # opt_prob.addCon('g1','i', lower=0., upper=1e21) # stress
    # opt_prob.addCon('g2','i', lower=0., upper=1e21) # displacement
    # # print(opt_prob)
    # snopt = pyOpt.pySNOPT.SNOPT()
    # snopt.setOption('Major iterations limit', maxiter)
    # snopt.setOption('Major optimality tolerance', tol)
    # snopt.setOption('Major feasibility tolerance', tol)
    # # snopt.setOption('Function precision', tol**2)
    # snopt(opt_prob, sens_type=pyopt_gradfunc, X=x, info=info)
    # status = int(opt_prob._solutions[0].opt_inform['value'])
    # while status in (3, 32): # 3:  accuracy could not be achieved
    #                          # 32: resource limit error (max iters)
    #     print("Re-solving optimization at random initial point since max iterations reached.")
    #     d0 = np.random.uniform(low=1.,high=4.,size=(2,))
    #     opt_prob._variables[0].value = d0[0]
    #     opt_prob._variables[1].value = d0[1]
    #     snopt(opt_prob, sens_type=pyopt_gradfunc, X=x, info=info)
    #     status = int(opt_prob._solutions[0].opt_inform['value'])
    # topt = opt_prob._solutions[0]._variables[0].value
    # wopt = opt_prob._solutions[0]._variables[1].value
    # success = True if status == 1 else False
    # fopt = opt_prob._solutions[0]._objectives[0].value
    # Result = collections.namedtuple('Result', 'x, fun, success')
    # result = Result(x=[topt,wopt], fun=fopt, success=success)

    result = minimize(obj, d0, args=(x), method='SLSQP', jac=objJac, 
        bounds=bounds, tol=tol, options={'maxiter': maxiter, 'disp': False},
        constraints=({'type': 'ineq', 'fun': sCon, 'jac': sConJac, 'args': (x,info)},
        {'type': 'ineq', 'fun': dCon, 'jac': dConJac, 'args': (x,info)}))
    while result.status == 9: # max iterations reached, restart from new point
        print("Re-solving optimization at random initial point since max iterations reached.")
        d0 = np.random.uniform(low=1.,high=4.,size=(2,))
        result = minimize(obj, d0, args=(x), method='SLSQP', jac=objJac, 
            bounds=bounds, tol=tol, options={'maxiter': maxiter, 'disp': False},
            constraints=({'type': 'ineq', 'fun': sCon, 'jac': sConJac, 'args': (x,info)},
            {'type': 'ineq', 'fun': dCon, 'jac': dConJac, 'args': (x,info)}))

    # Check results
    # print(result)

    return result

if __name__ == "__main__":

    # Setup optimization parameters
    d0 = [2.5, 2.5]
    X_mean = np.array([500., 1000., 2.9e7, 40000.]) # truth values
    X_stddev = np.array([100., 100., 1.45e6, 2000.]) # truth values
    m_samples = 30

    # Setup margin parameters
    info = {}
    # --- Data for constraints with basis values
    info["bv_p"] = 0.99
    info["bv_c"] = 0.95
    # --- Data for monte carlo estimates of reliability
    info["N"] = int(1e4) # number of samples
    info["pf"] = 0.00135 # desired probability of failure
    # --- Data for mean difference limit dispersion margin
    info["ldm_confidence_interval"] = 0.95

    # Run optimization for each formulation type
    f_list = ["det", "pi", "bv", "md_ldm", "mip_ldm"]
    for f in f_list:
        print(f)
        optimize(d0, X_mean, X_stddev, m_samples, formulation=f, info=info)