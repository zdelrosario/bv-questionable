### Uniaxial tension problem
# Comparison of different methods for managing statistical (sampling)
# uncertainty on a uniaxial tension test.
#
# Compares a number of approaches:
# BV         = basis value
# PI         = plug in
# MD LSSM    = mean difference limit state sample margin
#
# Zachary del Rosario, Mar. 2018

import numpy as np
import matplotlib.pyplot as plt
import time
import pyutil.numeric as ut
import pandas as pd

from scipy.stats import norm, nct, t, chi2
from scipy.optimize import bisect
from pyutil.plotting import linspecer

plt.style.use('ggplot')
np.random.seed(101)
np.set_printoptions(precision=3)

##################################################
## Script parameters
##################################################
# Full sampling
M     = int(1e3)           # Replications
N_ALL = [20,50,100,200,500,1000,2000] # Sample count sweep

# Select case
MYCASE = 2

# Design parameters
#                     Rel,  Con,  Pop
PARAM = np.array([[  0.90, 0.95, 0.99],    #  A-basis;  lax reliability
                  [  0.99, 0.95, 0.99],    #  A-basis;  med reliability
                  [1-1e-7, 0.95, 0.99],    #  A-basis;  strict reliability
                  [1-1e-7, 1-1e-7, 0.99]]) #  Z-basis;  strict reliability

# RV parameters
MU_CR  = 600.   * 1e+6 # mean, critical stress     [Pa]
MU_A   = 100.   * 1e+6 # mean, axial force         [N]
TAU_CR = MU_CR * 0.1  # std. dev., critical stress [Pa]
TAU_A  = MU_A  * 0.1  # std. dev., axial force     [N]
# Fixed DV
RADIUS = 1.0           # cylinder radius            [m]

D_LO = 1e-10; D_HI = RADIUS * 0.999 # thickness bounds for binary search [m]
DX   = np.sqrt(np.finfo(float).eps)

##################################################
# Helper functions
##################################################
fA_c   = lambda d: np.pi*(2*RADIUS*d + d**2)
fSig_a = lambda d: MU_A  / fA_c(d)
fGam   = lambda d: TAU_A / fA_c(d)

def fK_pc(p,c,n):
    return nct.ppf(c,n-1,-norm.ppf(1-p)*np.sqrt(n)) / np.sqrt(n)

def fZ_score(d,s_c,s):
    return (s_c - fSig_a(d)) / np.sqrt(s**2+fGam(d)**2)

def fRel_fcn(d,s_c,s):
    return norm.cdf(fZ_score(d,s_c,s))

def fRel_eff(d):
    return norm.cdf(fZ_score(d,MU_CR,TAU_CR))

def fD_bv(B,Rel):
    return np.sqrt((TAU_A*norm.ppf(Rel)+MU_A)/np.pi/B+RADIUS) - RADIUS

def fA_star(Rel,mu_cr=MU_CR,tau_cr=TAU_CR,M=0):
    z = norm.ppf(Rel)
    return ((mu_cr-M)*MU_A + np.sqrt(z**2*(mu_cr-M)**2*TAU_A**2 + z**2*MU_A**2*tau_cr**2 \
                        - z**4*tau_cr**2*TAU_A**2)) / ((mu_cr-M)**2 - z**2*tau_cr**2)

def fD_star(Rel,mu_cr=MU_CR,tau_cr=TAU_CR,M=0):
    A_star = fA_star(Rel,mu_cr=mu_cr,tau_cr=tau_cr,M=M)
    return np.sqrt(A_star/np.pi+RADIUS**2) - RADIUS

def g_lim(d,X):
    # Limit state function; g(-) > 0 for success
    # Usage
    #   g = g_lim(d,X)
    # Arguments
    #   d = Design variable
    #   X = Random variables;
    #     = [\sigma_cr, F_a]
    # Returns
    #   g = Limit state value

    return X[0] - X[1]/fA_c(d)

##################################################
# Design sweep
##################################################
Rel = PARAM[MYCASE][0]
Con = PARAM[MYCASE][1]
Pop = PARAM[MYCASE][2]

beta_cr = norm.ppf( Rel )
zc      = norm.ppf(Con)

d_s = fD_star(Rel)

D_lsm = np.zeros((len(N_ALL),M)); D_pi = np.zeros((len(N_ALL),M))
R_lsm = np.zeros((len(N_ALL),M)); R_pi = np.zeros((len(N_ALL),M))
M_lsm = np.zeros((len(N_ALL),M)); M_pi = np.zeros((len(N_ALL),M))
D_bv  = np.zeros((len(N_ALL),M)); D_psm = np.zeros((len(N_ALL),M))
R_bv  = np.zeros((len(N_ALL),M)); R_psm = np.zeros((len(N_ALL),M))
M_bv  = np.zeros((len(N_ALL),M)); M_psm = np.zeros((len(N_ALL),M))

M_samp = np.zeros((len(N_ALL),M))

t0 = time.time()
ut.print_progress(0,len(N_ALL)-1,bar_length=60)
for jnd in range(len(N_ALL)):
    for ind in range(M):
        X_dat = np.random.multivariate_normal(mean=[MU_CR,MU_A],
                                              cov=np.diag([TAU_CR**2,TAU_A**2]),
                                              size=N_ALL[jnd]).T
        mu_dat = np.mean(X_dat[0])
        s2_dat = np.var(X_dat[0])
        T_dat  = [mu_dat,s2_dat]

        ## Analytic BV approach
        B = mu_dat - fK_pc(Pop, Con, N_ALL[jnd]) * np.sqrt(s2_dat)
        D_bv[jnd,ind] = fD_bv(B,Rel)
        R_bv[jnd,ind] = fRel_eff(D_bv[jnd,ind])
        M_bv[jnd,ind] = (D_bv[jnd,ind]-d_s) / d_s

        ## Analytic PI approach
        D_pi[jnd,ind] = fD_star(Rel,mu_cr=mu_dat,tau_cr=np.sqrt(s2_dat))
        R_pi[jnd,ind] = fRel_eff(D_pi[jnd,ind])
        M_pi[jnd,ind] = (D_pi[jnd,ind]-d_s) / d_s

        ## Analytic PI + exact limit state margin
        m = zc*TAU_CR/np.sqrt(N_ALL[jnd])

        D_lsm[jnd,ind] = fD_star(Rel,mu_cr=mu_dat,tau_cr=np.sqrt(s2_dat),M=m)
        R_lsm[jnd,ind] = fRel_eff(D_lsm[jnd,ind])
        M_lsm[jnd,ind] = (D_lsm[jnd,ind]-d_s)/d_s

    ut.print_progress(jnd,len(N_ALL)-1,bar_length=60)
t1 = time.time()

##################################################
# Post-process
##################################################
D_lsm.sort(axis=1); D_pi.sort(axis=1); D_bv.sort(axis=1)
R_lsm.sort(axis=1); R_pi.sort(axis=1); R_bv.sort(axis=1)
M_lsm.sort(axis=1); M_pi.sort(axis=1); M_bv.sort(axis=1)
M_samp.sort(axis=1)

D_lsm_mu = np.mean(D_lsm,axis=1); D_pi_mu = np.mean(D_pi,axis=1)
R_lsm_mu = np.mean(R_lsm,axis=1); R_pi_mu = np.mean(R_pi,axis=1)
M_lsm_mu = np.mean(M_lsm,axis=1); M_pi_mu = np.mean(M_pi,axis=1)
D_bv_mu  = np.mean(D_bv,axis=1);  D_psm_mu  = np.mean(D_psm,axis=1)
R_bv_mu  = np.mean(R_bv,axis=1);  R_psm_mu  = np.mean(R_psm,axis=1)
M_bv_mu  = np.mean(M_bv,axis=1);  M_psm_mu  = np.mean(M_psm,axis=1)
M_samp_mu = np.mean(M_samp,axis=1)

# D_lsm_lo = D_lsm[:,int(M*((1-Con)/2))]; D_pi_lo = D_pi[:,int(M*((1-Con)/2))]
# R_lsm_lo = R_lsm[:,int(M*((1-Con)/2))]; R_pi_lo = R_pi[:,int(M*((1-Con)/2))]
# M_lsm_lo = M_lsm[:,int(M*((1-Con)/2))]; M_pi_lo = M_pi[:,int(M*((1-Con)/2))]
# D_bv_lo = D_bv[:,int(M*((1-Con)/2))];   D_psm_lo = D_psm[:,int(M*((1-Con)/2))]
# R_bv_lo = R_bv[:,int(M*((1-Con)/2))];   R_psm_lo = R_psm[:,int(M*((1-Con)/2))]
# M_bv_lo = M_bv[:,int(M*((1-Con)/2))];   M_psm_lo = M_psm[:,int(M*((1-Con)/2))]
# M_samp_lo = M_samp[:,int(M*((1-Con)/2))]

# D_lsm_hi = D_lsm[:,int(M*(1-(1-Con)/2))]; D_pi_hi = D_pi[:,int(M*(1-(1-Con)/2))]
# R_lsm_hi = R_lsm[:,int(M*(1-(1-Con)/2))]; R_pi_hi = R_pi[:,int(M*(1-(1-Con)/2))]
# M_lsm_hi = M_lsm[:,int(M*(1-(1-Con)/2))]; M_pi_hi = M_pi[:,int(M*(1-(1-Con)/2))]
# D_bv_hi = D_bv[:,int(M*(1-(1-Con)/2))];   D_psm_hi = D_psm[:,int(M*(1-(1-Con)/2))]
# R_bv_hi = R_bv[:,int(M*(1-(1-Con)/2))];   R_psm_hi = R_psm[:,int(M*(1-(1-Con)/2))]
# M_bv_hi = M_bv[:,int(M*(1-(1-Con)/2))];   M_psm_hi = M_psm[:,int(M*(1-(1-Con)/2))]
# M_samp_hi = M_samp[:,int(M*(1-(1-Con)/2))]

D_lsm_lo = D_lsm[:,int(M*((1-Con)))]; D_pi_lo = D_pi[:,int(M*((1-Con)))]
R_lsm_lo = R_lsm[:,int(M*((1-Con)))]; R_pi_lo = R_pi[:,int(M*((1-Con)))]
M_lsm_lo = M_lsm[:,int(M*((1-Con)))]; M_pi_lo = M_pi[:,int(M*((1-Con)))]
D_bv_lo = D_bv[:,int(M*((1-Con)))];   D_psm_lo = D_psm[:,int(M*((1-Con)))]
R_bv_lo = R_bv[:,int(M*((1-Con)))];   R_psm_lo = R_psm[:,int(M*((1-Con)))]
M_bv_lo = M_bv[:,int(M*((1-Con)))];   M_psm_lo = M_psm[:,int(M*((1-Con)))]
M_samp_lo = M_samp[:,int(M*((1-Con)))]

D_lsm_hi = D_lsm[:,int(M*(1-(1-Con)))]; D_pi_hi = D_pi[:,int(M*(1-(1-Con)))]
R_lsm_hi = R_lsm[:,int(M*(1-(1-Con)))]; R_pi_hi = R_pi[:,int(M*(1-(1-Con)))]
M_lsm_hi = M_lsm[:,int(M*(1-(1-Con)))]; M_pi_hi = M_pi[:,int(M*(1-(1-Con)))]
D_bv_hi = D_bv[:,int(M*(1-(1-Con)))];   D_psm_hi = D_psm[:,int(M*(1-(1-Con)))]
R_bv_hi = R_bv[:,int(M*(1-(1-Con)))];   R_psm_hi = R_psm[:,int(M*(1-(1-Con)))]
M_bv_hi = M_bv[:,int(M*(1-(1-Con)))];   M_psm_hi = M_psm[:,int(M*(1-(1-Con)))]
M_samp_hi = M_samp[:,int(M*(1-(1-Con)))]

##################################################
# Write results
##################################################
df = pd.DataFrame(
    data = {
        "N"       : N_ALL,
        "R"       : [Rel] * len(N_ALL),
        "C"       : [Con] * len(N_ALL),
        "P"       : [Pop] * len(N_ALL),

        "M_bv_mu" : M_bv_mu,
        "M_bv_lo" : M_bv_lo,
        "M_bv_hi" : M_bv_hi,

        "R_bv_mu" : R_bv_mu,
        "R_bv_lo" : R_bv_lo,
        "R_bv_hi" : R_bv_hi,

        "M_pi_mu" : M_pi_mu,
        "M_pi_lo" : M_pi_lo,
        "M_pi_hi" : M_pi_hi,

        "R_pi_mu" : R_pi_mu,
        "R_pi_lo" : R_pi_lo,
        "R_pi_hi" : R_pi_hi,

        "M_mil_mu" : M_lsm_mu,
        "M_mil_lo" : M_lsm_lo,
        "M_mil_hi" : M_lsm_hi,

        "R_mil_mu" : R_lsm_mu,
        "R_mil_lo" : R_lsm_lo,
        "R_mil_hi" : R_lsm_hi
    }
)

df.to_csv("../../data/an_mil_c{0:}.csv".format(MYCASE))

##################################################
# Report
##################################################
print("Execution time: {} sec".format(t1-t0))

Colors = linspecer(5)

### Effective margin
plt.figure()
## Data
# Basis Value
plt.plot(N_ALL,100*M_bv_lo,
         color = Colors[0,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL,100*M_bv_mu,
         color = Colors[0,:], linewidth = 2.0, label='AN + BV')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i]*0.95, N_ALL[i]*0.95],
             [100*M_bv_mu[i], 100*M_bv_lo[i]],
             color = Colors[0,:],
             linewidth = 0.5)

# Plug In
plt.plot(N_ALL,100*M_pi_lo,
         color = Colors[1,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL,100*M_pi_mu,
         color = Colors[1,:], linewidth = 2.0, label = 'AN + PI')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i], N_ALL[i]],
             [100*M_pi_mu[i], 100*M_pi_lo[i]],
             color = Colors[1,:],
             linewidth = 0.5)

# PI + exact limit state margin
plt.plot(N_ALL, 100*M_lsm_lo,
         color = Colors[3,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, 100*M_lsm_mu,
         color = Colors[3,:], linewidth = 2.0, label='AN + MIL')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i]*1.05, N_ALL[i]*1.05],
             [100*M_lsm_mu[i], 100*M_lsm_lo[i]],
             color = Colors[3,:],
             linewidth = 0.5)

## Annotation
plt.plot(N_ALL,[0]*len(N_ALL), 'k--', label='Requested')
plt.xlabel('Sample Count')
plt.ylabel('Effective Margin (%)')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
# Export
plt.savefig('../../images/an_mil_Meff_c{0:}.png'.format(MYCASE))
plt.close()

### Effective reliability
# Post-process: Failure probabilities
F_bv_lo = 1 - R_bv_lo; F_bv_hi = 1 - R_bv_hi; F_bv_mu = 1 - R_bv_mu
F_pi_lo = 1 - R_pi_lo; F_pi_hi = 1 - R_pi_hi; F_pi_mu = 1 - R_pi_mu
F_lsm_lo = 1 - R_lsm_lo; F_lsm_hi = 1 - R_lsm_hi; F_lsm_mu = 1 - R_lsm_mu

plt.figure()
## Data
# Basis Value
plt.plot(N_ALL, F_bv_lo,
         color = Colors[0,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, F_bv_mu,
         color = Colors[0,:], linewidth = 2.0, label = 'AN + BV')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i]*0.95, N_ALL[i]*0.95],
             [F_bv_mu[i], F_bv_lo[i]],
             color = Colors[0,:],
             linewidth = 0.5)

# Plug In
plt.plot(N_ALL, F_pi_lo,
         color = Colors[1,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, F_pi_mu,
         color = Colors[1,:], linewidth = 2.0, label = 'AN + PI')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i], N_ALL[i]],
             [F_pi_mu[i], F_pi_lo[i]],
             color = Colors[1,:],
             linewidth = 0.5)

# PI + exact limit state margin
plt.plot(N_ALL, F_lsm_lo,
         color = Colors[3,:], linewidth = 1.0, linestyle = ":")
plt.plot(N_ALL, F_lsm_mu,
         color = Colors[3,:], linewidth = 2.0, label = 'AN + MIL')
for i in range(len(N_ALL)):
    plt.plot([N_ALL[i]*1.05, N_ALL[i]*1.05],
             [F_lsm_mu[i], F_lsm_lo[i]],
             color = Colors[3,:],
             linewidth = 0.5)

## Annotation
plt.plot(N_ALL,[1-Rel]*len(N_ALL), 'k--',
         label = 'Requested', linewidth = 2.0)
plt.xlabel('Sample Count')
plt.ylabel('Effective Failure Chance (log Pr)')
plt.yscale('log')
plt.tight_layout()
plt.legend(loc=0)
axes = plt.gca()
axes.set_xlim([N_ALL[0],N_ALL[-1]])
plt.xscale('log')
# Export
plt.savefig('../../images/an_mil_Feff_c{0:}.png'.format(MYCASE))
plt.close()
