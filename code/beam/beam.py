"""
Subroutines used for the cantilevered beam problem originally found in:

Wu, Y.-T., Shin, Y., Sues, R., and Cesare, M., "Safety-factor based approach
for probability-based design optimization," American Institute of Aeronautics
and Astronautics, Seattle, Washington, 2001.

The functions in this file contain the following arguments:
    Arguments:
    d = list of length 2 with deterministic design variable values: [t, w]
        t = height of beam
        w = width of beam
    X = nx4 Numpy array of samples of random variables: [H, V, E, S] 
        H = lateral tip load
        V = vertical tip load
        E = Young's modulus of beam
        S = yield strength of beam

Zach del Rosario and Richard W. Fenrich, April 24, 2018
"""

import numpy as np 

L = 100. # in, length of beam
D0 = 2.2535 # in, max allowable deflection

def area(d, X):
    """ Returns t*w = cross-sectional area of beam"""
    t, w = d
    return t*w

def areaJac(d, X):
    """ Returns Jobj = Jacobian of cross-sectional area of beam"""
    t, w = d
    return np.array([w, t])

def stress(d, X):
    """ Returns gs = nx1 Numpy array, normalized stress limit state value 
        (<=0 is failure)"""
    t, w = d
    if len(X.shape) == 1: X = np.array([X])
    stress = 600.*X[:,1]/(w*t**2) + 600.*X[:,0]/(w**2*t)
    gs = 1 - stress/X[:,3] # normalized stress limit state (<= 0 is failure)
    return gs 

def stressJac(d, X):
    """ Returns Js = nx1 Numpy array, Jacobian for normalized stress limit state"""
    t, w = d
    if len(X.shape) == 1: X = np.array([X])
    dsdt = -1200*X[:,1]/(w*t**3) - 600*X[:,0]/(w**2*t**2)
    dsdw = -600*X[:,1]/(t**2*w**2) - 1200*X[:,0]/(t*w**3)
    dgsdt = (-1/X[:,3])*dsdt
    dgsdw = (-1/X[:,3])*dsdw
    return np.vstack((dgsdt,dgsdw)).T

def disp(d, X):
    """ Returns gd = nx1 Numpy array, normalized displacement limit state value 
        (<=0 is failure)"""
    t, w = d
    if len(X.shape) == 1: X = np.array([X])
    disp = 4*L**3/(X[:,2]*w*t)*np.sqrt((X[:,1]/t**2)**2 + (X[:,0]/w**2)**2)
    gd = 1 - disp/D0 # normalized displacement limit state (<= 0 is failure)
    return gd

def dispJac(d, X):
    """ Returns Jd = nx1 Numpy array, Jacobian for normalized displacment
    limit state"""
    t, w = d
    if len(X.shape) == 1: X = np.array([X])
    rt = np.sqrt((X[:,1]/t**2)**2 + (X[:,0]/w**2)**2)
    dddt = -4*L**3*rt/(X[:,2]*w*t**2) - 8*L**3*X[:,1]**2/(X[:,2]*t**6*w)/rt 
    dddw = -4*L**3*rt/(X[:,2]*t*w**2) - 8*L**3*X[:,0]**2/(X[:,2]*w**6*t)/rt
    dgddt = (-1/D0)*dddt 
    dgddw = (-1/D0)*dddw
    return np.vstack((dgddt, dgddw)).T

def deltaDFactor(X, mean, stddev):
    """
    Calculate multiplicative factor used for calculation of \nabla D_i in paper
    assuming all variables are normally distributed.
    Arguments:
    X = nx4 Numpy array of n samples of random variables: [H, V, E, S]
    mean = list of mean values of said 4 random variables
    stddev = list of std deviations of said 4 random variables
    Returns:
    factor = nx4 Numpy array with factor for each of 4 (unknown) distribution
        parameters and n samples. The distribution parameters are
        \theta = [\mu_E \sigma_E^2 \mu_S \sigma_S^2]
    """
    # For normally distributed R, \frac{\delta_{\theta} rho}{rho}
    n, m = X.shape
    factor = np.zeros((n,4))
    factor[:,0] = (X[:,2] - mean[2])/stddev[2]**2 # factor for /mu_E
    factor[:,1] = ((X[:,2] - mean[2])**2 - stddev[2]**2)/(2*stddev[2]**4) # factor for \sigma_E^2
    factor[:,2] = (X[:,3] - mean[3])/stddev[3]**2 # factor for /mu_S
    factor[:,3] = ((X[:,3] - mean[3])**2 - stddev[3]**2)/(2*stddev[3]**4) # factor for \sigma_S^2
    return factor

def covarianceDistributionParams(X_mean, X_stddev, m_samples):
    """ Return covariance matrix for (unknown) distribution parameters. In this
    case E and S from X = [H, V, E, S]. """
    cov = np.diag([X_stddev[2]**2/m_samples, X_stddev[2]**4/(m_samples-1), 
        X_stddev[3]**2/m_samples, X_stddev[3]**4/(m_samples-1)])
    return cov